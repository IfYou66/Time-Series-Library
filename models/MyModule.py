import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """
    使用FFT检测时间序列的主要周期
    核心思想：通过频域分析找到时间序列的主要频率成分

    参数：
        x: 输入张量 [B, T, C]
        k: 返回前k个主要周期

    返回：
        period: 主要周期列表
        period_weight: 对应的权重
    """
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = xf.abs().mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)      # [k]
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list                  # [k]
    return period, xf.abs().mean(-1)[:, top_list]    # [B, k]


class TimesBlock(nn.Module):
    """
    TimesNet核心模块：1D→2D转换并用2D卷积提取多周期特征
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, C = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            p = int(period_list[i])
            L = self.seq_len + self.pred_len
            if L % p != 0:
                L_pad = ((L // p) + 1) * p
                pad = torch.zeros(B, L_pad - L, C, device=x.device)
                out = torch.cat([x, pad], dim=1)
            else:
                L_pad = L
                out = x

            # 1D→2D reshape & permute to [B, C, num_periods, p]
            out = out.reshape(B, L_pad // p, p, C).permute(0, 3, 1, 2)
            out = self.conv(out)  # [B, d_model, num_periods, p]

            # 2D→1D back to [B, L_pad, C]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, C)
            res.append(out[:, :L, :])

        # stack along last dim → [B, L, C, k]
        res = torch.stack(res, dim=-1)
        # weight & aggregate
        w = F.softmax(period_weight, dim=1)         # [B, k]
        w = w.unsqueeze(1).unsqueeze(1).repeat(1, T, C, 1)
        res = (res * w).sum(dim=-1)                 # [B, T, C]
        return res + x                              # residual


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs   = configs
        self.task_name = configs.task_name
        # 分类时固定类别数
        self.num_class = getattr(configs, 'num_class', None)

        # 1) TimesBlock 堆叠
        self.model = nn.ModuleList(
            [TimesBlock(configs) for _ in range(configs.e_layers)]
        )
        # 2) 数据嵌入
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq,
            configs.dropout
        )
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 3) 任务专属头
        if self.task_name in ('long_term_forecast','short_term_forecast'):
            self.predict_linear = nn.Linear(
                configs.seq_len, configs.seq_len + configs.pred_len
            )
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True
            )

        elif self.task_name in ('imputation','anomaly_detection'):
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True
            )

        elif self.task_name == 'classification':
            assert self.num_class is not None, "configs.num_class must be set for classification"
            # TapNet 风格注意力原型头
            self.att_dim = getattr(configs, 'att_dim', 128)
            self.att_models = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(configs.d_model, self.att_dim),
                    nn.Tanh(),
                    nn.Linear(self.att_dim, 1)
                )
                for _ in range(self.num_class)
            ])
            # 可选激活 & Dropout
            self.act     = F.gelu
            self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_labels=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ('long_term_forecast','short_term_forecast'):
            out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return out[:, -self.configs.pred_len:, :]

        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)

        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc, x_labels)

        return None

    def classification(self, x_enc, x_mark_enc, labels):
        """
        TapNet 风格注意力原型分类：
         1) embed + TimesBlock → [B, T, D]
         2) mask 掩掉 padding
         3) 全局 avg pool → [B, D]
         4) per-class attention 加权求 prototype → [num_class, D]
         5) cdist→ logits
        """
        # 1) embed + 特征提取
        enc_out = self.enc_embedding(x_enc, None)   # [B, T, D]
        for blk in self.model:
            enc_out = self.layer_norm(blk(enc_out))

        # 2) mask 掩掉 padding
        mask = x_mark_enc.unsqueeze(-1)             # [B, T, 1]
        enc_out = enc_out * mask

        # 3) 全局 avg pool → x_emb: [B, D]
        lengths = mask.sum(dim=1).clamp(min=1)      # [B, 1]
        x_emb = enc_out.sum(dim=1) / lengths        # [B, D]

        # 4) 构造每类 prototype
        proto_list = []
        D = x_emb.size(-1)
        for k in range(self.num_class):
            idx_k = (labels == k).nonzero().squeeze(1)
            if idx_k.numel() == 0:
                proto_list.append(torch.zeros(D, device=x_emb.device))
            else:
                h_k = x_emb[idx_k]                   # [N_k, D]
                a_k = self.att_models[k](h_k).squeeze(1)  # [N_k]
                a_k = F.softmax(a_k, dim=0)             # [N_k]
                c_k = (h_k * a_k.unsqueeze(1)).sum(dim=0)  # [D]
                proto_list.append(c_k)
        prototypes = torch.stack(proto_list, dim=0)    # [num_class, D]

        # 5) 计算距离并取负号为 logits
        dists  = torch.cdist(x_emb, prototypes, p=2)   # [B, num_class]
        logits = -dists

        return logits
