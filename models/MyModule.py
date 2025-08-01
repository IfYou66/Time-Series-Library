import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ------------------------------------------------------------
# 1) 序列分解模块：SeriesDecomp（不变）
# ------------------------------------------------------------
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, count_include_pad=False
        )
    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        trend = self.avg_pool(x.permute(0,2,1)).permute(0,2,1)  # [B, T, C]
        seasonal = x - trend
        return seasonal, trend

# ------------------------------------------------------------
# 2) per-sample / per-head cross-autocorrelation
# ------------------------------------------------------------
def cross_auto_correlation(x: torch.Tensor, k: int, n_heads: int):
    """
    x: [B, T, C]
    returns:
        periods: [B, n_heads, k]  top-k lags per sample per head (ints)
        weights: [B, n_heads, k]  softmaxed scores per sample per head
    """
    B, T, C = x.shape
    assert C % n_heads == 0, "d_model must be divisible by n_heads"
    head_dim = C // n_heads
    # reshape into heads: [B, n_heads, T, head_dim]
    x_heads = x.reshape(B, T, n_heads, head_dim).permute(0,2,1,3)  # [B, n_heads, T, head_dim]

    # FFT along time dim
    Xf = torch.fft.rfft(x_heads, dim=2)  # [B, n_heads, F, head_dim]
    P = Xf * torch.conj(Xf)              # power spectrum [B, n_heads, F, head_dim]
    # average over head_dim to get per-head power
    Pm = P.mean(-1)                      # [B, n_heads, F]
    # irFFT to get autocorrelation per head per sample
    r = torch.fft.irfft(Pm, n=T, dim=2)  # [B, n_heads, T]
    # zero out lag 0 to avoid trivial
    r = r.clone()
    r[..., 0] = 0.0
    # top-k along lag dimension
    vals, idx = torch.topk(r, k, dim=2)  # both [B, n_heads, k]
    weights = F.softmax(vals, dim=2)      # softmax over the k lags
    return idx, weights  # periods, weights

# ------------------------------------------------------------
# 3) TimesBlock（周期对齐 + 2D-Conv + 趋势残差）——重写版
# ------------------------------------------------------------
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len    = configs.seq_len
        self.pred_len   = configs.pred_len
        self.k          = configs.top_k
        self.kernel     = configs.moving_avg
        self.n_heads    = getattr(configs, 'n_heads', 1)
        # 分解
        self.decomp     = SeriesDecomp(self.kernel)
        # 原始 Inception 2D（保持结构，但它现在接受合并后的通道）
        self.conv2d     = nn.Sequential(
            nn.Conv2d(configs.d_model, configs.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(configs.d_model, configs.d_model, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        B, T, C = x.shape
        total = self.seq_len if self.pred_len == 0 else self.seq_len + self.pred_len  # 目标长度 L
        # 1) 分解（基于原始输入长度 T）
        s_orig, t_orig = self.decomp(x)  # [B, T, C] each

        # 2) 计算 per-sample / per-head cross-autocorrelation（基于 seasonal 原始部分）
        periods, weights = cross_auto_correlation(s_orig, self.k, self.n_heads)
        # periods, weights: [B, n_heads, k]

        # 3) 为 forecast 需要 pad seasonal/trend/residual 到 total 长度
        if total > T:
            pad_len = total - T
            # replicate last time step
            s = F.pad(s_orig, (0,0,0,pad_len), mode='replicate')  # [B, total, C]
            t = F.pad(t_orig, (0,0,0,pad_len), mode='replicate')  # [B, total, C]
            x_resid = F.pad(x, (0,0,0,pad_len), mode='replicate')  # residual for skip connection
        else:
            s = s_orig
            t = t_orig
            x_resid = x

        L = total  # shorthand

        # 4) 以 head 维度处理 seasonal: [B, L, C] -> [B, n_heads, L, head_dim]
        assert C % self.n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = C // self.n_heads
        s_heads = s.reshape(B, L, self.n_heads, head_dim).permute(0,2,1,3)  # [B, n_heads, L, head_dim]

        # 5) 利用 periods 生成 rolled 版本（向量化处理所有 lag）
        # periods: [B, n_heads, k]
        # Prepare time indices
        device = x.device
        base = torch.arange(L, device=device)  # [L]
        # shifted_indices: [B, n_heads, k, L] where each entry is (t + p) % L with p being lag
        shifted_indices = (base[None,None,None,:] + periods.unsqueeze(-1)) % L  # [B, n_heads, k, L]
        # gather to get rolled seasonal per lag: expand for head_dim
        # s_heads: [B, n_heads, L, head_dim] -> unsqueeze to match
        expanded = s_heads.unsqueeze(2).expand(B, self.n_heads, self.k, L, head_dim)  # [B, n_heads, k, L, head_dim]
        gather_idx = shifted_indices.unsqueeze(-1).expand(B, self.n_heads, self.k, L, head_dim)  # same shape
        rolled = torch.gather(expanded, 3, gather_idx)  # [B, n_heads, k, L, head_dim]

        # 6) 将 head 和 lag 维度合并，送入 2D conv
        # 首先恢复每个 slice 到原始维度再做 conv2d per (B,lag) by merging head into channel
        # 合并 head 和 head_dim 形成 channel=C, 保持 k 在 batch 维度
        # rolled: [B, n_heads, k, L, head_dim] -> [B, k, L, C]
        rolled = rolled.permute(0,2,3,1,4).reshape(B, self.k, L, C)  # combine heads & head_dim
        # 为 conv2d 变形: 需要 [B * k, C, cycles, p] 风格的 2D 表示
        # 这里我们重新划分 L 为 2D grid: 为了尽量保留原意，选一个近似 reshape：
        # 令 p = periods（取每 sample/head 的第一个 head-averaged lag）做近似，实际 2D 维度用 (ceil(L / p), p)
        # 为避免循环，用每个 lag 的最大 p 统一 reshape（简化处理）
        # 取各 sample 对应 k 个 lag 中的最大 lag 作为 p_max
        p_max = periods.max().item()  # scalar
        # 计算 cycles = ceil(L / p_max)
        cycles = math.ceil(L / p_max)
        full_len = cycles * p_max
        pad_len_2d = full_len - L  # scalar
        # pad temporal to full_len for all
        rolled_padded = F.pad(rolled, (0,0,0,pad_len_2d), mode='replicate')  # [B, k, full_len, C]
        # reshape to 2D grid: (cycles, p_max)
        rolled_2d = rolled_padded.reshape(B * self.k, cycles, p_max, C)  # [B*k, cycles, p_max, C]
        # move channel to front for conv2d: expecting [B*k, C, H, W]
        rolled_2d = rolled_2d.permute(0,3,1,2)  # [B*k, C, cycles, p_max]
        # apply shared 2D conv
        conv_out = self.conv2d(rolled_2d)  # [B*k, C, cycles, p_max]
        # revert shape back: [B, k, full_len, C]
        conv_out = conv_out.permute(0,2,3,1).reshape(B, self.k, cycles * p_max, C)  # [B, k, full_len, C]
        # crop back to length L
        y = conv_out[:, :, :L, :]  # [B, k, L, C]

        # 7) 恢复 head 结构（这里把 head 聚合后的 C 用权重加权）
        # weights currently [B, n_heads, k],我们在 head 上平均以得到每 lag 的权重
        # 也可以做更精细的重组合（如 head-wise 但此处简化）
        weights_lag = weights.mean(1)  # [B, k]
        weights_lag = weights_lag.unsqueeze(-1).unsqueeze(-1)  # [B, k, 1, 1]
        weighted = y * weights_lag  # [B, k, L, C]
        agg = weighted.sum(1)  # sum over k -> [B, L, C]

        # 8) 加回 trend（已经 pad 到 L）和残差
        out = agg + t  # [B, L, C]
        out = out + x_resid  # skip connection, [B, L, C]
        return out

# ------------------------------------------------------------
# 4) 分类模型：简单投影 + N×TimesBlock + 池化 + 分类头（略微加了 assert）
# ------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 输入特征→d_model
        self.project = nn.Linear(configs.enc_in, configs.d_model)
        # 核心块
        self.blocks  = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])
        # 每层归一化
        self.norms   = nn.ModuleList([
            nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)
        ])
        # 池化和分类
        self.dropout    = nn.Dropout(configs.dropout)
        self.classifier = nn.Linear(configs.d_model, configs.num_class)

        assert configs.d_model % getattr(configs, 'n_heads', 1) == 0, "d_model must be divisible by n_heads"

    def forward(self, x_enc, *args):
        """
        x_enc: [B, T, enc_in]
        """
        # 1) 特征投影
        x = self.project(x_enc)           # [B, T, d_model]
        # 2) N 层 TimesBlock + Norm
        for blk, ln in zip(self.blocks, self.norms):
            x = ln(blk(x))                # [B, T, d_model]
        # 3) 全局平均池化
        x = x.mean(dim=1)                 # [B, d_model]
        x = self.dropout(x)
        # 4) 分类输出
        out = self.classifier(x)          # [B, num_class]
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
