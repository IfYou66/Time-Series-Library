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
    # [B, T, C] - 批次大小、时间步长、特征维度
    xf = torch.fft.rfft(x, dim=1)  # 对时间维度进行FFT变换
    # 通过振幅找到主要周期
    frequency_list = abs(xf).mean(0).mean(-1)  # 计算每个频率的平均振幅
    frequency_list[0] = 0  # 将直流分量（频率0）设为0
    _, top_list = torch.topk(frequency_list, k)  # 找到前k个最大振幅的频率
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list  # 计算对应的周期长度
    return period, abs(xf).mean(-1)[:, top_list]  # 返回周期和对应的权重


class TimesBlock(nn.Module):
    """
    TimesNet的核心模块：将1D时间序列转换为2D表示进行处理

    核心思想：
    1. 通过FFT检测时间序列的主要周期
    2. 将1D时间序列重塑为2D表示（时间-周期）
    3. 使用2D卷积捕获时间-周期维度的模式
    4. 自适应聚合不同周期的结果
    """

    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len  # 输入序列长度
        self.pred_len = configs.pred_len  # 预测序列长度
        self.k = configs.top_k  # 使用前k个主要周期

        # 参数高效设计：使用Inception块进行2D卷积
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),  # 第一个Inception块
            nn.GELU(),  # 激活函数
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)  # 第二个Inception块
        )

    def forward(self, x):
        """
        TimesBlock的前向传播
        关键步骤：
        1. FFT周期检测
        2. 1D到2D转换
        3. 2D卷积处理
        4. 2D到1D转换
        5. 自适应聚合
        6. 残差连接
        """
        B, T, N = x.size()  # 批次大小、时间步长、特征维度

        # 1. FFT周期检测：找到时间序列的主要周期
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]  # 当前处理的周期

            # 2. 填充处理：确保序列长度能被周期整除
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)  # 添加填充
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # 3. 1D到2D转换：重塑为(批次, 特征, 周期数, 周期长度)
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # 4. 2D卷积：从1D变化转换为2D变化处理
            out = self.conv(out)

            # 5. 2D到1D转换：重塑回原始格式
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        # 6. 堆叠不同周期的结果
        res = torch.stack(res, dim=-1)

        # 7. 自适应聚合：使用softmax权重聚合不同周期的结果
        period_weight = F.softmax(period_weight, dim=1)  # 归一化权重
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)  # 扩展维度
        res = torch.sum(res * period_weight, -1)  # 加权求和

        # 8. 残差连接：保持梯度流动
        res = res + x
        return res


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs    = configs
        self.task_name  = configs.task_name
        # 1. TimesBlock 层
        self.model      = nn.ModuleList(
            [TimesBlock(configs) for _ in range(configs.e_layers)]
        )
        # 2. 数据嵌入
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq,
            configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        # 3. 任务专属头
        if self.task_name in ('long_term_forecast','short_term_forecast'):
            self.predict_linear = nn.Linear(
                configs.seq_len, configs.seq_len+configs.pred_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

        elif self.task_name in ('imputation','anomaly_detection'):
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

        elif self.task_name == 'classification':
            # --- 原型学习头 ---
            # 注意力隐藏维度
            self.att_dim = getattr(configs, 'att_dim', 128)
            # 每类一个小 MLP，产生注意力分数
            self.att_models = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(configs.d_model, self.att_dim),
                    nn.Tanh(),
                    nn.Linear(self.att_dim, 1)
                )
                for _ in range(configs.num_class)
            ])
            # 不再使用原来的 flatten+projection
            # 但仍保留 act & dropout 用于 embedding 后处理
            self.act     = F.gelu
            self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_labels=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ('long_term_forecast','short_term_forecast'):
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.configs.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            # x_labels: Tensor (B,)  —— 在 Exp_Classification.train/vali/test 时，
            # 需要把 label 传入此位置（即 model(batch_x, padding_mask, label)）
            return self.classification(x_enc, x_mark_enc, x_labels)
        return None

    def classification(self, x_enc, x_mark_enc, labels):
        """
        用 TapNet 的注意力原型方法替代原先的平铺+线性：
         1) embed→(B,T,D)
         2) mask→zero-pad
         3) 时序 avg pool→(B,D)
         4) per-class attention 加权求原型→(C,D)
         5) 样本到原型的欧氏距离→logits
        """
        # 1. embed + TimesBlock
        enc_out = self.enc_embedding(x_enc, None)  # [B, T, D]
        for blk in self.model:
            enc_out = self.layer_norm(blk(enc_out))
        # 2. mask 掩掉 padding
        mask = x_mark_enc.unsqueeze(-1)             # [B, T, 1]
        enc_out = enc_out * mask

        # 3. 按时间维做 avg pooling 得到 embedding
        lengths = mask.sum(dim=1).clamp(min=1)      # [B, 1]
        x_emb = enc_out.sum(dim=1) / lengths        # [B, D]

        # 4. 构造每个类别的原型
        proto_list = []
        C, D = labels.max().item() + 1, x_emb.size(-1)
        for k in range(C):
            idx_k = (labels == k).nonzero().squeeze(1)
            if idx_k.numel() == 0:
                # 如果该batch没有某类样本，则用 0 向量代替
                proto_list.append(torch.zeros(D, device=x_emb.device))
            else:
                h_k = x_emb[idx_k]                  # [N_k, D]
                # 4.1 打分→softmax
                a_k = self.att_models[k](h_k)      # [N_k, 1]
                a_k = F.softmax(a_k.squeeze(1), dim=0)  # [N_k]
                # 4.2 加权求和
                c_k = (h_k * a_k.unsqueeze(1)).sum(dim=0)  # [D]
                proto_list.append(c_k)
        prototypes = torch.stack(proto_list, dim=0)  # [C, D]

        # 5. 计算样本到所有原型的欧氏距离，并取负值作 logits
        # 用 torch.cdist（也可用手写 pairwise）
        dists = torch.cdist(x_emb, prototypes, p=2)  # [B, C]
        logits = -dists

        return logits
