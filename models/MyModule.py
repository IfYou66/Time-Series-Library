# ------------------------------------------------------------
# version4 best
# 加入环形对齐模块 CircularConvAlongP（在周期维 p 上做 depthwise 1×k 的 circular padding 卷积）→ 提升对相位漂移的鲁棒性而无需显式 roll。
# 将 Conv2dResBlock 升级为轻量 Inception 多尺度块（3×3、5×5、dilated‑3×3 的并行深度可分离卷积 + 1×1 融合 + 残差），并用 GroupNorm 提升小批次稳定性。
# 高效分支聚合：按 unique(p) 进行分桶，对每个 p 一次性处理所有 (b, j) 选择，用 index_add_ 实现对 agg_num/agg_den 的张量化累加，去掉内层 j 循环。
# 反射填充改为通道优先安全实现（[B,C,T] 上 ReflectionPad1d），并对 auto_correlation 做了无效候选回退，避免 NaN。
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# =========================
# 工具：安全的 GroupNorm 组数选择（保证可整除）
# =========================
def _best_gn_groups(C: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if C % g == 0:
            return g
    return 1

# =========================
# 残差缩放（LayerScale 风格）
# =========================
class ResidualScale(nn.Module):
    def __init__(self, channels: int, init: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * init)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma

# =========================
# 1) 序列分解
# =========================
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, count_include_pad=False
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,T,C]
        x_nct = x.permute(0, 2, 1)                     # [B,C,T]
        trend = self.avg_pool(x_nct).permute(0, 2, 1)  # [B,T,C]
        seasonal = x - trend
        return seasonal, trend


# =========================
# 2) 自相关（稳健回退）
# =========================
def auto_correlation(x: torch.Tensor, k: int, min_p: int = 3, max_p: int = None):
    """
    x: [B, T, C]
    return:
      idx: [B, k]   每个样本的周期滞后
      w  : [B, k]   自相关峰值 softmax 权重（先验）
    """
    B, T, C = x.shape
    dtype = x.dtype
    if max_p is None:
        max_p = max(1, T // 2)
    max_p = max(min_p, min(max_p, T - 1))

    Xf = torch.fft.rfft(x, dim=1)                  # [B,F,C]
    P = (Xf * torch.conj(Xf)).real                 # [B,F,C]
    Pm = P.mean(dim=-1)                            # [B,F]（如需更强可做可学习通道加权）
    r = torch.fft.irfft(Pm, n=T, dim=1)            # [B,T]

    # 合法掩码
    mask = torch.ones_like(r, dtype=torch.bool)
    mask[:, 0] = False
    if min_p > 1:
        mask[:, 1:min_p] = False
    if max_p + 1 < T:
        mask[:, max_p + 1:] = False

    very_neg = torch.finfo(dtype).min / 8 if dtype.is_floating_point else -1e9
    r_masked = torch.where(mask, r, torch.full_like(r, very_neg))

    # 若全非法，回退到 p0
    p0 = max(min_p, min(max_p, max(2, T // 4)))
    all_bad = (mask.sum(dim=1) == 0)
    if all_bad.any():
        r_masked[all_bad] = very_neg
        r_masked[all_bad, p0] = 0.0

    k_eff = min(k, max_p - min_p + 1)
    vals, idx = torch.topk(r_masked, k=k_eff, dim=1)
    if k_eff < k:
        pad_n = k - k_eff
        idx = torch.cat([idx, idx[:, -1:].expand(B, pad_n)], dim=1)
        vals = torch.cat([vals, vals[:, -1:].expand(B, pad_n)], dim=1)
    w = F.softmax(vals, dim=1)
    return idx, w


# =========================
# 3) 工具：安全反射折叠/展开
# =========================
def _fold_2d_reflect(x_1d: torch.Tensor, p: int) -> Tuple[torch.Tensor, int]:
    """
    x_1d: [b,T,C] → z: [b,C,cyc,p], T_orig
    """
    b, T, C = x_1d.shape
    pad = ((T + p - 1) // p) * p - T
    if pad > 0:
        x_nct = x_1d.permute(0, 2, 1).contiguous()     # [b,C,T]
        x_nct = F.pad(x_nct, (0, pad), mode='reflect') # 仅右侧填充
        x_1d  = x_nct.permute(0, 2, 1).contiguous()    # [b,T+pad,C]
    T_new = x_1d.shape[1]
    cyc = T_new // p
    z = x_1d.reshape(b, cyc, p, C).permute(0, 3, 1, 2).contiguous()  # [b,C,cyc,p]
    return z, T


def _unfold_1d(z_2d: torch.Tensor, T: int) -> torch.Tensor:
    """
    z_2d: [b,C,cyc,p] → y: [b,T,C]
    """
    b, C, cyc, p = z_2d.shape
    y = z_2d.permute(0, 2, 3, 1).reshape(b, cyc * p, C)[:, :T, :].contiguous()
    return y


# =========================
# 4) SE（通道注意力）
# =========================
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(reduction, channels // 4)
        g = _best_gn_groups(channels)
        # 用 Conv2d 实现 SE，保持轻量
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),              # [B,C,1,1]
            nn.Conv2d(channels, hidden, 1, bias=False), nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False), nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,C,cyc,p]
        w = self.fc(z)
        return z * w


# =========================
# 5) SK 融合（支持温度/先验）
# =========================
class SKMerge(nn.Module):
    def __init__(self, channels: int, n_branches: int = 3, reduction: int = 8, tau: float = 1.5):
        super().__init__()
        hidden = max(reduction, channels // 4)
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, n_branches * channels, 1, bias=False),
        )
        self.n = n_branches
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 频谱先验权重（如需）
        self.tau = tau                                # softmax 温度：>1 更均匀，<1 更尖锐

    def forward(self, feats, w_prior: torch.Tensor = None):
        """
        feats: List of [B,C,cyc,p], len = n
        w_prior: None 或 [B,1,1,1] / [B, n, 1, 1] / [B,n,C,1,1]
        """
        B, C, _, _ = feats[0].shape
        s = sum(feats) / self.n                         # [B,C,cyc,p]
        a = self.squeeze(s).view(B, self.n, C, 1, 1)    # [B,n,C,1,1]
        if w_prior is not None:
            # 广播到 [B,n,C,1,1]
            if w_prior.dim() == 4:      # [B,1,1,1]
                w_prior = w_prior.view(B, 1, 1, 1, 1).expand(B, self.n, C, 1, 1)
            elif w_prior.dim() == 5 and w_prior.size(1) == 1:  # [B,1,1,1,1]
                w_prior = w_prior.expand(B, self.n, C, 1, 1)
            a = a + self.alpha * w_prior
        # 温度软化
        w = torch.softmax(a / self.tau, dim=1)          # [B,n,C,1,1]
        out = sum(w[:, i] * feats[i] for i in range(self.n))   # [B,C,cyc,p]
        return out


# =========================
# 6) Inception 2D 残差块（多尺度 + SK + SE）+ Pre-Norm + 残差缩放
# =========================
class Inception2dResBlock_SKSE(nn.Module):
    def __init__(self, channels: int, drop: float = 0.0, sk_tau: float = 1.5, se_strength: float = 0.0):
        super().__init__()
        C = channels
        g = _best_gn_groups(C)

        # Pre-Norm
        self.pre = nn.GroupNorm(num_groups=g, num_channels=C)

        # 三个深度可分离分支
        self.b3   = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=C), nn.GELU()
        )
        self.b5   = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=5, padding=2, groups=C, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=C), nn.GELU()
        )
        self.bDil = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=2, dilation=2, groups=C, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=C), nn.GELU()
        )
        self.sk   = SKMerge(C, n_branches=3, reduction=max(8, C // 4), tau=sk_tau)
        self.merge_pw = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.merge_gn = nn.GroupNorm(num_groups=g, num_channels=C)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        # 第二阶段轻量卷积
        self.dw   = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
        self.pw   = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.gn2  = nn.GroupNorm(num_groups=g, num_channels=C)
        self.act  = nn.GELU()

        # 通道 SE 门控（强度可控）
        self.se = SEBlock(C, reduction=max(8, C // 4))
        self.se_strength = se_strength

        # 残差缩放
        self.res_scale = ResidualScale(C, init=1e-3)

    def forward(self, z: torch.Tensor, w_prior_branch: torch.Tensor = None) -> torch.Tensor:
        # z: [B,C,cyc,p]
        x_in = self.pre(z)                                # Pre-Norm
        f1, f2, f3 = self.b3(x_in), self.b5(x_in), self.bDil(x_in)
        x = self.sk([f1, f2, f3], w_prior=w_prior_branch) # [B,C,cyc,p]
        x = self.merge_pw(x)
        x = self.merge_gn(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.dw(x)
        x = self.pw(x)
        x = self.gn2(x)
        x = self.act(x)

        if self.se_strength > 0:
            # 将 SE 当成缓和因子：pow(se_strength)
            x = x * (self.se(x).pow(self.se_strength))

        return z + self.res_scale(x)                      # 缩放后的残差


# =========================
# 7) cyc 轴大核 Conv1d（动态有效核长 + 3x堆叠近似 + 轻量门控 + 残差缩放）
# =========================
class CycLargeKernelConv1d(nn.Module):
    """
    在 [B,C,cyc,p] 的特征上，先对 p 轴做聚合得到 [B,C,cyc]，
    然后沿 cyc 轴做深度可分离 1D 大核卷积（用多次3x卷积近似），再广播回 2D。
    """
    def __init__(self, channels: int, k: int = 15):
        super().__init__()
        C = channels
        self.k_cfg = k
        g = _best_gn_groups(C)
        # 用 3x depthwise 近似大核，autotune 更稳
        self.dw3 = nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
        self.pw  = nn.Conv1d(C, C, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=g, num_channels=C)
        self.act  = nn.GELU()
        # 轻量门控，避免覆盖原特征
        self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(C, C, 1, bias=False), nn.Sigmoid())
        # 残差缩放
        self.res_scale = ResidualScale(C, init=1e-3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,C,cyc,p]
        u = z.mean(dim=-1)                               # [B,C,cyc]
        cyc = u.size(-1)
        if cyc <= 1:
            return z  # 边界保护

        # 有效核长不超过 2*cyc-1，且不小于3
        k_eff = max(3, min(self.k_cfg, 2 * cyc - 1))
        rep = max(1, (k_eff // 3))

        x = u
        for _ in range(rep):
            x = self.dw3(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)

        g = self.gate(u)                                 # [B,C,1]
        x = x * g                                        # 轻量门控
        x = x.unsqueeze(-1).expand_as(z)                 # [B,C,cyc,p]
        return z + self.res_scale(x)                     # 缩放后的残差


# =========================
# 8) TimesBlock（集成：SK + cyc-Conv1d + SE）
# =========================
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.k        = configs.top_k
        self.kernel   = configs.moving_avg
        self.d_model  = configs.d_model
        self.min_p    = getattr(configs, 'min_period', 3)
        self.drop2d   = getattr(configs, 'conv2d_dropout', 0.0)
        self.cyc_k    = getattr(configs, 'cyc_conv_kernel', 9)   # 起步用 9 更稳
        self.sk_tau   = getattr(configs, 'sk_tau', 1.5)          # 初期更均匀
        self.se_strength = getattr(configs, 'se_strength', 0.0)  # 先关/弱开

        self.decomp   = SeriesDecomp(self.kernel)
        self.block2d  = Inception2dResBlock_SKSE(self.d_model, drop=self.drop2d,
                                                 sk_tau=self.sk_tau, se_strength=self.se_strength)
        self.cyc1d    = CycLargeKernelConv1d(self.d_model, k=self.cyc_k)

        # 内容感知门控（样本级标量，用于最终融合）
        hidden = max(8, self.d_model // 4)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,C=d_model]
        """
        B, T, C = x.shape
        assert C == self.d_model, "Input last dim must equal d_model"

        # 分解
        s, t = self.decomp(x)  # [B,T,C]

        # 候选周期与先验
        idx, w_prior = auto_correlation(s, self.k, min_p=self.min_p, max_p=T // 2)  # [B,k],[B,k]

        # 聚合容器
        agg_num = torch.zeros_like(x)    # [B,T,C]
        agg_den = x.new_zeros(B)         # [B]

        # 以 unique(p) 分桶
        unique_p = torch.unique(idx)
        for pv in unique_p.tolist():
            mask = (idx == pv)                          # [B,k]
            if not mask.any():
                continue
            b_idx, j_idx = mask.nonzero(as_tuple=True)  # [m],[m]
            m = b_idx.numel()

            sb = s[b_idx]                                # [m,T,C]
            wb = w_prior[b_idx, j_idx].view(m, 1, 1)     # [m,1,1]

            # 折叠
            z, _T = _fold_2d_reflect(sb, int(pv))        # [m,C,cyc,p]

            # 2D 多尺度 + SK + SE（含 Pre-Norm/残差缩放）
            z = self.block2d(z)                          # [m,C,cyc,p]

            # cyc 轴大核 Conv1d（跨周期长程，含残差缩放/门控/动态有效核长）
            z = self.cyc1d(z)                            # [m,C,cyc,p]

            # 展回 1D
            y = _unfold_1d(z, _T)                        # [m,T,C]

            # 样本级内容门控
            gate = torch.sigmoid(self.gate_mlp(y.mean(dim=1)))   # [m,1]
            score = torch.log(wb + self.eps) + torch.log(gate.view(m, 1, 1) + self.eps)
            score_exp = torch.exp(score)                 # [m,1,1]

            contrib = y * score_exp                      # [m,T,C]
            agg_num.index_add_(0, b_idx, contrib)
            agg_den.index_add_(0, b_idx, score_exp.view(-1))

        # 归一化融合
        agg = agg_num / (agg_den.view(B, 1, 1) + self.eps)

        # 加回趋势，并保留季节残差
        out = agg + t
        return out + (x - t)


# =========================
# 9) 分类模型（与原接口一致）
# =========================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.d_model   = configs.d_model
        self.num_class = configs.num_class
        self.dropout_p = configs.dropout

        self.project = nn.Linear(configs.enc_in, self.d_model)
        self.blocks  = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.norms   = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(configs.e_layers)])

        # 注意力池化（可学习 query）
        self.pool_q = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.dropout    = nn.Dropout(self.dropout_p)
        self.classifier = nn.Linear(self.d_model, self.num_class)

    def attention_pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,d] → [B,d]
        B, T, d = x.shape
        q = self.pool_q.expand(B, -1, -1)                       # [B,1,d]
        att = torch.matmul(q, x.transpose(1, 2)) / (d ** 0.5)   # [B,1,T]
        att = torch.softmax(att, dim=-1)
        return torch.matmul(att, x).squeeze(1)                  # [B,d]

    def forward(self, x_enc: torch.Tensor, *args) -> torch.Tensor:
        x = self.project(x_enc)                                  # [B,T,d]
        for blk, ln in zip(self.blocks, self.norms):
            x = ln(blk(x))                                       # [B,T,d]
        x = self.attention_pool(x)                               # [B,d]
        x = self.dropout(x)
        out = self.classifier(x)                                 # [B,num_class]
        return out

