# model/timesnet_plus.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

# ------------------------------------------------------------
# 1) 序列分解模块：SeriesDecomp
# ------------------------------------------------------------
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super(SeriesDecomp, self).__init__()
        self.kernel_size = kernel_size
        # 1D 移动平均，padding='same'
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, count_include_pad=False
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, C]
        return:
          x_seasonal: [B, T, C], 去趋势后的季节成分
          x_trend:    [B, T, C], 平滑出的趋势成分
        """
        # 转到 [B, C, T] 做 1D 平均池化
        x_perm = x.permute(0, 2, 1)  # [B, C, T]
        trend = self.avg_pool(x_perm)  # [B, C, T]
        trend = trend.permute(0, 2, 1)  # [B, T, C]
        seasonal = x - trend
        return seasonal, trend

# ------------------------------------------------------------
# 2) 自相关函数：Auto-Correlation
# ------------------------------------------------------------
def auto_correlation(x: torch.Tensor, k: int):
    """
    基于 Wiener–Khinchin：通过频谱反变换一次计算所有滞后自相关，
    再 Top-k 选出主周期，并按 softmax 归一化权重。
    x: [B, T, C]
    return:
      periods: List[int]  长度=k 的滞后列表
      weights: Tensor    [k]  对应的 softmax 权重
    """
    B, T, C = x.shape
    # FFT 获得频谱
    Xf = torch.fft.rfft(x, dim=1)            # [B, F, C]
    P = Xf * torch.conj(Xf)                  # [B, F, C]
    P_mean = P.mean(0).mean(-1)              # [F]
    # 反变换到时域，得自相关 r(τ) for τ=0..T-1
    r = torch.fft.irfft(P_mean, n=T, dim=0)  # [T]
    r[0] = 0  # 忽略 0 滞后
    # Top-k 滞后
    vals, idx = torch.topk(r, k)            # each scalar
    weights = F.softmax(vals, dim=0)         # [k]
    periods = idx.tolist()                  # 转成 Python list
    return periods, weights

# ------------------------------------------------------------
# 3) TimesBlock（融合 SeriesDecomp、Auto-Corr、2D PosEncoding）
# ------------------------------------------------------------
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k                   # Top-k 周期数
        self.kernel_size = configs.moving_avg  # 分解移动平均窗口

        # 3.1 内置分解
        self.series_decomp = SeriesDecomp(self.kernel_size)

        # 3.2 2D 可学习位置编码（行：周期索引，列：相位索引）
        #    允许最大周期数=seq_len, 最大周期长度=seq_len
        self.pos_col = nn.Parameter(torch.randn(1, 1, configs.seq_len, 1))
        self.pos_row = nn.Parameter(torch.randn(1, 1, 1, configs.seq_len))

        # 3.3 2D 卷积模块（原 Inception）
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, C]
        """
        B, T, C = x.shape

        # —— 1) 内置分解：先提取趋势并剥离季节
        x_seasonal, x_trend = self.series_decomp(x)  # [B,T,C] x2

        # —— 2) 自相关找周期
        periods, weights = auto_correlation(x_seasonal, self.k)
        # weights: [k]

        outs = []
        for i, p in enumerate(periods):
            # —— (a) roll 对齐：将季节信号按周期对齐
            rolled = torch.roll(x_seasonal, -p, dims=1)

            # —— (b) pad & reshape to 2D：->[B, C, cycles, p]
            total_len = self.seq_len + self.pred_len
            if total_len % p != 0:
                pad_len = ( (total_len // p + 1)*p - total_len )
                padded = F.pad(rolled, (0,0,0,pad_len))  # pad 时间维度前
            else:
                padded = rolled
            cycles = padded.shape[1] // p
            out2d = padded.reshape(B, cycles, p, C).permute(0,3,1,2).contiguous()  # [B,C,cycles,p]

            # —— (c) 加二维位置编码
            #    pos_col: [1,1,max_cycles,1], pos_row: [1,1,1,max_p]
            out2d = out2d + self.pos_col[:,:,:cycles,:] + self.pos_row[:,:,:,:p]

            # —— (d) 2D 卷积提取
            conv2d = self.conv(out2d)  # [B,C,cycles,p]

            # —— (e) reshape 回 1D
            back = conv2d.permute(0,2,3,1).reshape(B, cycles*p, C)
            outs.append(back[:, :total_len, :])  # 裁到 [B, total_len, C]

        # —— 3) 多周期聚合：按权重加权
        stack = torch.stack(outs, dim=-1)          # [B, L, C, k]
        w = weights.view(1,1,1,self.k).to(x.device)
        agg = (stack * w).sum(-1)                  # [B, L, C]

        # —— 4) 加回趋势 & 残差
        #      4.1 加回趋势部分
        out = agg + x_trend
        #      4.2 残差连接
        out = out + x

        return out

# ------------------------------------------------------------
# 4) 主模型：Model（仅 classification 部分显示关键插入）
# ------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # 嵌入层
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        # N 层 TimesBlock
        self.blocks = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 分类头
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len,
                configs.num_class
            )

    def classification(self, x_enc, x_mask):
        """
        x_enc: [B, T, C], x_mask: [B, T]（padding mask）
        """
        # 1) 嵌入
        enc = self.enc_embedding(x_enc, None)  # 时间标记对分类可选
        # 2) N 层 TimesBlock
        for blk in self.blocks:
            enc = self.layer_norm(blk(enc))
        # 3) 非线性 + Dropout + Mask
        out = self.act(enc)
        out = self.dropout(out)
        out = out * x_mask.unsqueeze(-1)
        # 4) 展平 & 分类
        out = out.reshape(out.shape[0], -1)
        out = self.projection(out)
        return out

    def forward(self, x_enc, x_mask, *args):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mask)
        else:
            raise NotImplementedError(f"Task {self.task_name} not supported in this variant.")

