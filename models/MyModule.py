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
        x: 输入张量 [B, T, C] - 批次大小、时间步长、特征维度
        k: 返回前k个主要周期

    返回：
        period: 主要周期列表
        period_weight: 对应的权重
    """
    xf = torch.fft.rfft(x, dim=1)  # 对时间维度进行FFT变换
    frequency_list = xf.abs().mean(0).mean(-1)  # 计算每个频率的平均振幅
    frequency_list[0] = 0  # 将直流分量（频率0）设为0
    _, top_list = torch.topk(frequency_list, k)  # 找到前k个最大振幅的频率
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list  # 计算对应的周期长度
    return period, xf.abs().mean(-1)[:, top_list]  # 返回周期和对应的权重


class TimesBlock(nn.Module):
    """
    TimesNet的核心模块：将1D时间序列转换为2D表示进行处理，
    并在2D→1D聚合后加入SE通道注意力
    """
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len  = configs.seq_len
        self.pred_len = configs.pred_len
        self.k        = configs.top_k

        # 2D 多尺度卷积
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

        # SE 通道注意力
        C = configs.d_model
        r = getattr(configs, 'se_ratio', 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),         # [B, C, 1]
            nn.Conv1d(C, C // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(C // r, C, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        1) FFT周期检测
        2) 1D→2D转换 & 2D卷积
        3) 2D→1D转换 & 周期加权聚合
        4) SE通道注意力
        5) 残差连接
        """
        B, T, C_in = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        outs = []
        L = self.seq_len + self.pred_len
        for i in range(self.k):
            p = int(period_list[i])
            if L % p != 0:
                Lp = ((L // p) + 1) * p
                pad = torch.zeros(B, Lp - L, C_in, device=x.device)
                xi = torch.cat([x, pad], dim=1)
            else:
                Lp = L
                xi = x

            # 1D→2D reshape to [B, C_in, num_periods, p]
            xi = xi.reshape(B, Lp // p, p, C_in).permute(0, 3, 1, 2).contiguous()
            xi = self.conv(xi)
            # 2D→1D reshape back to [B, Lp, C_in]
            xi = xi.permute(0, 2, 3, 1).reshape(B, -1, C_in)
            outs.append(xi[:, :L, :])

        # stack & period-weighted sum → [B, T, C_in]
        out = torch.stack(outs, dim=-1)                 # [B, T, C_in, k]
        w   = F.softmax(period_weight, dim=1)           # [B, k]
        w   = w.unsqueeze(1).unsqueeze(1).repeat(1, T, C_in, 1)
        out = (out * w).sum(dim=-1)                     # [B, T, C_in]

        # SE 通道注意力
        out_ca    = out.permute(0, 2, 1)                # [B, C_in, T]
        se_w      = self.se(out_ca)                    # [B, C_in, 1]
        out_ca    = out_ca * se_w                       # [B, C_in, T]
        out       = out_ca.permute(0, 2, 1)             # [B, T, C_in]

        # 残差连接
        return out + x


class Model(nn.Module):
    """
    TimesNet主模型
    支持：long_term_forecast, short_term_forecast,
          imputation, anomaly_detection, classification
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs   = configs
        self.task_name = configs.task_name
        self.seq_len   = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len  = configs.pred_len

        # 1. 核心TimesBlock层
        self.model = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])

        # 2. 数据嵌入
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq,
            configs.dropout
        )
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 3. 任务专属Head
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len
            )
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True
            )

        if self.task_name in ('imputation', 'anomaly_detection'):
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True
            )

        if self.task_name == 'classification':
            self.act       = F.gelu
            self.dropout   = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len,
                configs.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # ——— 与原版完全相同 ———
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0,2,1)).permute(0,2,1)
        for blk in self.model:
            enc_out = self.layer_norm(blk(enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:,0,:].unsqueeze(1).repeat(1,self.pred_len+self.seq_len,1))
        dec_out = dec_out.add(means[:,0,:].unsqueeze(1).repeat(1,self.pred_len+self.seq_len,1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # ——— 与原版完全相同 ———
        means = torch.sum(x_enc, dim=1) / torch.sum(mask==1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means).masked_fill(mask==0, 0)
        stdev = torch.sqrt(torch.sum(x_enc*x_enc, dim=1)/torch.sum(mask==1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for blk in self.model:
            enc_out = self.layer_norm(blk(enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:,0,:].unsqueeze(1).repeat(1,self.pred_len+self.seq_len,1))
        dec_out = dec_out.add(means[:,0,:].unsqueeze(1).repeat(1,self.pred_len+self.seq_len,1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # ——— 与原版完全相同 ———
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, None)
        for blk in self.model:
            enc_out = self.layer_norm(blk(enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:,0,:].unsqueeze(1).repeat(1,self.pred_len+self.seq_len,1))
        dec_out = dec_out.add(means[:,0,:].unsqueeze(1).repeat(1,self.pred_len+self.seq_len,1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # ——— 与原版完全相同 ———
        enc_out = self.enc_embedding(x_enc, None)
        for blk in self.model:
            enc_out = self.layer_norm(blk(enc_out))
        out = self.act(enc_out)
        out = self.dropout(out)
        out = out * x_mark_enc.unsqueeze(-1)
        out = out.reshape(out.size(0), -1)
        out = self.projection(out)
        return out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ('long_term_forecast','short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
