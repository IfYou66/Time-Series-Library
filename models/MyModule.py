import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ------------------------------------------------------------
# 1) 序列分解：SeriesDecomp（中心移动平均）
# ------------------------------------------------------------
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, count_include_pad=False
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, C]
        return: seasonal, trend  (same shape as x)
        """
        trend = self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B,T,C]
        seasonal = x - trend
        return seasonal, trend


# ------------------------------------------------------------
# 2) 自相关（逐样本）：Auto-Correlation
#    只用当前长度 T；返回每个样本的 top-k 周期与权重
# ------------------------------------------------------------
def auto_correlation(x: torch.Tensor, k: int):
    """
    x: [B, T, C]
    return:
      idx: [B, k]   每个样本的周期(滞后)
      w  : [B, k]   对应 softmax 权重
    """
    B, T, C = x.shape
    Xf = torch.fft.rfft(x, dim=1)             # [B, F, C]
    P = (Xf * torch.conj(Xf)).real            # [B, F, C]
    Pm = P.mean(dim=-1)                       # [B, F]  # 逐样本功率谱
    r = torch.fft.irfft(Pm, n=T, dim=1)       # [B, T]  # 逐样本自相关
    r[:, 0] = float('-inf')                   # 禁止零滞后
    vals, idx = torch.topk(r, k, dim=1)       # [B, k]
    w = F.softmax(vals, dim=1)                # [B, k]
    return idx, w


# ------------------------------------------------------------
# 3) 轻量 2D 卷积块：Conv-BN-GELU-Conv-BN + 残差
# ------------------------------------------------------------
class Conv2dResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.block(z)               # 残差


# ------------------------------------------------------------
# 4) TimesBlock（逐样本周期 + 只用 T + 反射填充 + 2D 残差块）
# ------------------------------------------------------------
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.k       = configs.top_k
        self.kernel  = configs.moving_avg
        self.d_model = configs.d_model

        self.decomp  = SeriesDecomp(self.kernel)
        self.conv2d  = Conv2dResBlock(self.d_model)

    @staticmethod
    def _fold_2d(x_1d: torch.Tensor, p: int) -> torch.Tensor:
        """
        x_1d: [b, T, C]
        p:    period
        return z: [b, C, cyc, p]
        """
        b, T, C = x_1d.shape
        pad = ((T + p - 1) // p) * p - T
        if pad > 0:
            # 反射填充在时间维（dim=1）
            x_1d = F.pad(x_1d, (0, 0, 0, pad), mode='reflect')  # (C_pad=0, T_pad=pad)
        cyc = (T + pad) // p
        z = x_1d.reshape(b, cyc, p, C).permute(0, 3, 1, 2)      # [b,C,cyc,p]
        return z, T  # 返回原始 T 便于后续裁剪

    @staticmethod
    def _unfold_1d(z_2d: torch.Tensor, T: int) -> torch.Tensor:
        """
        z_2d: [b, C, cyc, p]
        return y: [b, T, C]
        """
        b, C, cyc, p = z_2d.shape
        y = z_2d.permute(0, 2, 3, 1).reshape(b, cyc * p, C)[:, :T, :]  # [b,T,C]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C=d_model]
        """
        B, T, C = x.shape
        assert C == self.d_model, "Input last dim must equal d_model"

        # 分解
        s, t = self.decomp(x)  # [B,T,C]
        # 逐样本周期与权重
        idx, w = auto_correlation(s, self.k)  # [B,k], [B,k]

        # 聚合容器（始终使用当前 T）
        agg = torch.zeros_like(x)

        # 按第 j 个候选周期累加
        for j in range(self.k):
            p_b = idx[:, j]            # [B]
            # 将相同 p 的样本分组以做并行
            unique_p = p_b.unique()
            for pv in unique_p.tolist():
                sel = (p_b == pv)
                if not torch.any(sel):
                    continue
                sb = s[sel]                                # [b,T,C]
                wb = w[sel, j].view(-1, 1, 1)              # [b,1,1]
                # 对齐（向前滚动 pv）
                rolled = torch.roll(sb, shifts=-int(pv), dims=1)
                # 折叠 -> 2D
                z, _T = self._fold_2d(rolled, int(pv))     # [b,C,cyc,p]
                # 2D 残差卷积
                z = self.conv2d(z)                         # [b,C,cyc,p]
                # 展回 1D 并裁剪到 T
                y = self._unfold_1d(z, _T)                 # [b,T,C]
                # 权重加权并累加
                agg[sel] = agg[sel] + y * wb

        # 加回趋势 + 残差
        out = agg + t
        return out + x


# ------------------------------------------------------------
# 5) 分类模型：投影 + N×TimesBlock + 注意力池化 + 分类头
#    （用注意力池化替换原来的 GAP）
# ------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.d_model   = configs.d_model
        self.num_class = configs.num_class
        self.dropout_p = configs.dropout

        # 输入特征 -> d_model
        self.project = nn.Linear(configs.enc_in, self.d_model)

        # 堆叠 TimesBlock
        self.blocks = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])

        # Post-Norm（保持你原先用法；如果不稳可切 Pre-Norm）
        self.norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(configs.e_layers)])

        # 注意力池化
        self.pool_q = nn.Parameter(torch.randn(1, 1, self.d_model))

        # 分类头
        self.dropout    = nn.Dropout(self.dropout_p)
        self.classifier = nn.Linear(self.d_model, self.num_class)

    def attention_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d]
        return: [B, d]
        """
        B, T, d = x.shape
        q = self.pool_q.expand(B, -1, -1)                  # [B,1,d]
        att = torch.matmul(q, x.transpose(1, 2)) / (d ** 0.5)  # [B,1,T]
        att = torch.softmax(att, dim=-1)
        x_pooled = torch.matmul(att, x).squeeze(1)         # [B,d]
        return x_pooled

    def forward(self, x_enc: torch.Tensor, *args) -> torch.Tensor:
        """
        x_enc: [B, T, enc_in]
        """
        # 投影
        x = self.project(x_enc)                            # [B,T,d]
        # N 层 TimesBlock
        for blk, ln in zip(self.blocks, self.norms):
            x = ln(blk(x))                                 # [B,T,d]
        # 注意力池化
        x = self.attention_pool(x)                         # [B,d]
        x = self.dropout(x)
        # 分类
        out = self.classifier(x)                           # [B,num_class]
        return out


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # ------------------------------------------------------------
# # 1) 序列分解模块：SeriesDecomp
# # ------------------------------------------------------------
# class SeriesDecomp(nn.Module):
#     def __init__(self, kernel_size: int):
#         super().__init__()
#         self.avg_pool = nn.AvgPool1d(
#             kernel_size=kernel_size, stride=1,
#             padding=(kernel_size - 1) // 2, count_include_pad=False
#         )
#     def forward(self, x: torch.Tensor):
#         # x: [B, T, C]
#         # 转 [B,C,T] 做移动平均
#         trend = self.avg_pool(x.permute(0,2,1)).permute(0,2,1)  # [B, T, C]
#         seasonal = x - trend
#         return seasonal, trend
#
# # ------------------------------------------------------------
# # 2) 自相关函数：Auto-Correlation
# # ------------------------------------------------------------
# def auto_correlation(x: torch.Tensor, k: int):
#     # x: [B, T, C]
#     B, T, C = x.shape
#     # FFT→功率谱
#     Xf = torch.fft.rfft(x, dim=1)              # [B, F, C]
#     P  = Xf * torch.conj(Xf)                   # [B, F, C]
#     Pm = P.mean(0).mean(-1)                    # [F]
#     # irFFT→自相关序列 r(τ)
#     r = torch.fft.irfft(Pm, n=T, dim=0)        # [T]
#     r[0] = 0
#     vals, idx = torch.topk(r, k)               # 选 Top‐k 滞后
#     w = F.softmax(vals, dim=0)                 # [k]
#     return idx.tolist(), w                     # periods, weights
#
# # ------------------------------------------------------------
# # 3) TimesBlock（周期对齐 + 2D-Conv + 趋势残差）
# # ------------------------------------------------------------
# class TimesBlock(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.seq_len    = configs.seq_len
#         self.pred_len   = configs.pred_len
#         self.k          = configs.top_k
#         self.kernel     = configs.moving_avg
#         # 分解 & Inception 2D
#         self.decomp     = SeriesDecomp(self.kernel)
#         self.conv2d     = nn.Sequential(
#             # 这里用原 Inception_Block_V1
#             nn.Conv2d(configs.d_model, configs.d_model, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(configs.d_model, configs.d_model, kernel_size=3, padding=1),
#         )
#
#     def forward(self, x: torch.Tensor):
#         # x: [B, T, C]
#         B, T, C = x.shape
#         # 1) 分解
#         s, t = self.decomp(x)                   # seasonal, trend
#         # 2) 自相关选周期
#         periods, weights = auto_correlation(s, self.k)
#         outs = []
#         total = self.seq_len + self.pred_len
#         for p, w in zip(periods, weights):
#             # roll 对齐
#             rolled = torch.roll(s, -p, dims=1)
#             # pad & reshape→2D: [B,C,cycles,p]
#             L = total
#             if L % p != 0:
#                 pad = ( (L//p+1)*p - L )
#                 rolled = F.pad(rolled, (0,0,0,pad))
#             cyc = rolled.shape[1] // p
#             z = rolled.reshape(B, cyc, p, C).permute(0,3,1,2)  # [B,C,cyc,p]
#             # 2D conv
#             z = self.conv2d(z)
#             # reshape回1D
#             y = z.permute(0,2,3,1).reshape(B, cyc*p, C)[:,:L,:]
#             outs.append(y * w)  # 直接加权
#         # 聚合 & 加回趋势 + 残差
#         agg = torch.stack(outs, dim=0).sum(0)      # [B,L,C]
#         out = agg + t
#         return out + x
#
# # ------------------------------------------------------------
# # 4) 分类模型：简单投影 + N×TimesBlock + 池化 + 分类头
# # ------------------------------------------------------------
# class Model(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         # 输入特征→d_model
#         self.project = nn.Linear(configs.enc_in, configs.d_model)
#         # 核心块
#         self.blocks  = nn.ModuleList([
#             TimesBlock(configs) for _ in range(configs.e_layers)
#         ])
#         # 每层归一化
#         self.norms   = nn.ModuleList([
#             nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)
#         ])
#         # 池化和分类
#         self.dropout    = nn.Dropout(configs.dropout)
#         self.classifier = nn.Linear(configs.d_model, configs.num_class)
#
#     def forward(self, x_enc, *args):
#         """
#         x_enc: [B, T, enc_in]
#         """
#         # 1) 特征投影
#         x = self.project(x_enc)           # [B, T, d_model]
#         # 2) N 层 TimesBlock
#         for blk, ln in zip(self.blocks, self.norms):
#             x = ln(blk(x))                # [B, T, d_model]
#         # 3) 全局平均池化
#         x = x.mean(dim=1)                 # [B, d_model]
#         x = self.dropout(x)
#         # 4) 分类输出
#         out = self.classifier(x)          # [B, num_class]
#         return out
