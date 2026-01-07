import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config (SE 已彻底删除)
# ============================================================

@dataclass
class DSDNetConfig:
    # Data / task
    enc_in: int
    num_class: int

    # Backbone dims
    d_model: int = 128
    e_layers: int = 3
    dropout: float = 0.1

    # Period detection / decomposition
    top_k: int = 3
    moving_avg: int = 25
    use_series_decomp: bool = True
    min_period: int = 3
    max_period: Optional[int] = None  # default: T//2 at runtime

    # 2D conv trunk
    conv2d_dropout: float = 0.0
    use_sk: bool = True
    sk_tau: float = 1.5
    use_res_scale: bool = True
    reflect_pad: bool = True

    # cyc-axis modeling
    use_cyc_conv1d: bool = True
    cyc_conv_kernel: int = 9  # fixed kernel for speed

    # Weighted aggregation
    use_gate_mlp: bool = True  # sample-wise scalar gate for aggregation

    # Dual-path (non-periodic)
    use_dual_path: bool = True
    local_kernel: int = 7
    use_global_attn: bool = False  # simplified attn vs global pooling gate

    # Fusion options
    use_orthogonal: bool = True         # orthogonalize local residual w.r.t periodic residual
    use_spa_gate: bool = True           # SPA bias gate with period confidence
    gate_alpha: float = 4.0             # SPA bias strength
    gate_tau: float = 0.5               # SPA threshold
    dual_beta: float = 0.3              # local residual amplitude cap
    gate_warmup_epochs: int = 0         # optionally freeze bypass early (0 disables)

    # Normalization in 2D trunk (important for unified grid)
    use_masked_gn2d: bool = True        # keep semantics closer under unified grid

    # Pooling head
    use_attention_pool: bool = True     # learnable query pooling


def build_cfg_from_args(args) -> DSDNetConfig:
    """
    将 TimesNet 仓库的 argparse.Namespace 映射到 DSDNetConfig
    注意：SE 已彻底移除，此处不再读取 use_se/se_strength
    """
    cfg = DSDNetConfig(
        enc_in=getattr(args, "enc_in"),
        num_class=getattr(args, "num_class"),

        d_model=getattr(args, "d_model", 128),
        e_layers=getattr(args, "e_layers", 3),
        dropout=getattr(args, "dropout", 0.1),

        top_k=getattr(args, "top_k", 3),
        moving_avg=getattr(args, "moving_avg", 25),
        use_series_decomp=bool(getattr(args, "use_series_decomp", 1)),
        min_period=getattr(args, "min_period", 3),
        max_period=getattr(args, "max_period", None),

        conv2d_dropout=getattr(args, "conv2d_dropout", 0.0),
        use_sk=bool(getattr(args, "use_sk", 1)),
        sk_tau=float(getattr(args, "sk_tau", 1.5)),
        use_res_scale=bool(getattr(args, "use_res_scale", 1)),
        reflect_pad=bool(getattr(args, "reflect_pad", 1)),

        use_cyc_conv1d=bool(getattr(args, "use_cyc_conv1d", 1)),
        cyc_conv_kernel=int(getattr(args, "cyc_conv_kernel", 9)),

        use_gate_mlp=bool(getattr(args, "use_gate_mlp", 1)),

        use_dual_path=bool(getattr(args, "use_dual_path", 0)),
        local_kernel=int(getattr(args, "local_kernel", 7)),
        use_global_attn=bool(getattr(args, "use_global_attn", 0)),

        use_orthogonal=True,
        use_spa_gate=True,
        gate_alpha=float(getattr(args, "gate_alpha", 4.0)),
        gate_tau=float(getattr(args, "gate_tau", 0.5)),
        dual_beta=float(getattr(args, "dual_beta", 0.3)),
        gate_warmup_epochs=int(getattr(args, "gate_warmup", 0)),  # argparse: gate_warmup

        use_masked_gn2d=True,
        use_attention_pool=True,
    )
    return cfg


# ============================================================
# Utilities
# ============================================================

def _best_gn_groups(C: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if C % g == 0:
            return g
    return 1


class ResidualScale2D(nn.Module):
    """LayerScale for 4D: [B,C,H,W]"""
    def __init__(self, channels: int, init: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class ResidualScale1D(nn.Module):
    """LayerScale for 3D: [B,C,T]"""
    def __init__(self, channels: int, init: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1) * init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


def _period_confidence_from_wprior(w_prior: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    w_prior: [B,k], softmax weights for top-k peaks
    conf in [0,1], closer to 1 => clearer periodicity (lower entropy)
    """
    H = -(w_prior * (w_prior + eps).log()).sum(dim=1)  # [B]
    k = w_prior.size(1)
    if k <= 1:
        return torch.zeros_like(H)
    H_max = torch.log(torch.tensor(float(k), device=w_prior.device, dtype=w_prior.dtype) + eps)
    conf = 1.0 - H / (H_max + eps)
    return conf.clamp(0.0, 1.0)


def _orth_residual(local_res: torch.Tensor, periodic_res: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    local_res, periodic_res: [B,T,C]
    local' = local - proj(local on periodic)
    """
    num = (local_res * periodic_res).sum(dim=1, keepdim=True)              # [B,1,C]
    den = (periodic_res * periodic_res).sum(dim=1, keepdim=True).add(eps)  # [B,1,C]
    proj = (num / den) * periodic_res                                     # [B,T,C]
    return local_res - proj


def _reflect_indices(t: torch.Tensor, T: int) -> torch.Tensor:
    """
    Right-side reflect padding indices for length T sequence.
    t: [...], integer tensor
    Return idx in [0, T-1]
    """
    if T <= 1:
        return torch.zeros_like(t)
    m = 2 * (T - 1)
    r = t % m
    idx = torch.where(r < T, r, m - r)
    return idx


# ============================================================
# Decomposition
# ============================================================

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2,
            count_include_pad=False
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,T,C]
        x_nct = x.permute(0, 2, 1)                     # [B,C,T]
        trend = self.avg_pool(x_nct).permute(0, 2, 1)  # [B,T,C]
        seasonal = x - trend
        return seasonal, trend


# ============================================================
# Period detection (FFT-based autocorrelation)
# ============================================================

def auto_correlation_topk(
    x: torch.Tensor,
    k: int,
    min_p: int = 3,
    max_p: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [B,T,C]
    returns:
      idx: [B,k] (period lags)
      w  : [B,k] (softmax weights from correlation peaks)
    """
    B, T, C = x.shape
    dtype = x.dtype
    if max_p is None:
        max_p = max(1, T // 2)
    max_p = max(min_p, min(max_p, T - 1))

    Xf = torch.fft.rfft(x, dim=1)                 # [B,F,C]
    P = (Xf * torch.conj(Xf)).real                # [B,F,C]
    Pm = P.mean(dim=-1)                           # [B,F]
    r = torch.fft.irfft(Pm, n=T, dim=1)           # [B,T]

    mask = torch.ones_like(r, dtype=torch.bool)
    mask[:, 0] = False
    if min_p > 1:
        mask[:, 1:min_p] = False
    if max_p + 1 < T:
        mask[:, max_p + 1:] = False

    very_neg = torch.finfo(dtype).min / 8 if dtype.is_floating_point else -1e9
    r_masked = torch.where(mask, r, torch.full_like(r, very_neg))

    # fallback if all invalid (rare)
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


# ============================================================
# Masked GroupNorm for 2D
# ============================================================

class MaskedGroupNorm2d(nn.Module):
    """
    GroupNorm variant for [B,C,H,W] with spatial mask [B,1,H,W] (0/1).
    Computes per-sample per-group mean/var over masked positions only.
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        assert num_channels % num_groups == 0
        self.G = num_groups
        self.C = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,C,H,W], mask: [B,1,H,W]
        B, C, H, W = x.shape
        G = self.G
        xg = x.view(B, G, C // G, H, W)

        if mask is None:
            mean = xg.mean(dim=(2, 3, 4), keepdim=True)
            var = xg.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        else:
            mg = mask.view(B, 1, 1, H, W)
            denom = mg.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
            mean = (xg * mg).sum(dim=(2, 3, 4), keepdim=True) / denom
            var = ((xg - mean) ** 2 * mg).sum(dim=(2, 3, 4), keepdim=True) / denom

        xg = (xg - mean) / torch.sqrt(var + self.eps)
        y = xg.view(B, C, H, W)

        if self.affine:
            y = y * self.weight + self.bias
        return y


# ============================================================
# SK (SE 已删除)
# ============================================================

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
        self.tau = tau

    def forward(self, feats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        B, C, _, _ = feats[0].shape
        s = sum(feats) / self.n
        a = self.squeeze(s).view(B, self.n, C, 1, 1)
        w = torch.softmax(a / self.tau, dim=1)
        out = 0.0
        for i in range(self.n):
            out = out + w[:, i] * feats[i]
        return out


# ============================================================
# Periodic trunk: Inception 2D residual block (SK) + masked GN
# ============================================================

class Inception2DResBlock(nn.Module):
    """
    Multi-scale periodic feature extractor on folded 2D plane.
    Works on [B,C,H(=cyc),W(=p_max)] with mask.
    """
    def __init__(
        self,
        channels: int,
        use_sk: bool = True,
        sk_tau: float = 1.5,
        use_res_scale: bool = True,
        drop: float = 0.0,
        use_masked_gn: bool = True
    ):
        super().__init__()
        C = channels
        g = _best_gn_groups(C)
        self.use_masked_gn = use_masked_gn

        self.pre_gn = MaskedGroupNorm2d(g, C) if use_masked_gn else nn.GroupNorm(g, C)

        def _gn():
            return MaskedGroupNorm2d(g, C) if use_masked_gn else nn.GroupNorm(g, C)

        self.b3 = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            _gn(), nn.GELU()
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(C, C, 5, padding=2, groups=C, bias=False),
            _gn(), nn.GELU()
        )
        self.bDil = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=2, dilation=2, groups=C, bias=False),
            _gn(), nn.GELU()
        )

        self.use_sk = bool(use_sk)
        self.sk = SKMerge(C, n_branches=3, reduction=max(8, C // 4), tau=sk_tau) if self.use_sk else None

        self.merge_pw = nn.Conv2d(C, C, 1, bias=False)
        self.merge_gn = _gn()
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.dw = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.pw = nn.Conv2d(C, C, 1, bias=False)
        self.gn2 = _gn()

        self.use_res_scale = bool(use_res_scale)
        self.res_scale = ResidualScale2D(C, init=1e-3) if self.use_res_scale else nn.Identity()

    def _apply_norm(self, layer: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if isinstance(layer, MaskedGroupNorm2d):
            return layer(x, mask)
        return layer(x)

    def forward(self, z: torch.Tensor, mask2d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z: [B,C,H,W]
        mask2d: [B,1,H,W], float {0,1}
        """
        x = self._apply_norm(self.pre_gn, z, mask2d)

        def run_branch(seq: nn.Sequential, x_in: torch.Tensor) -> torch.Tensor:
            y = seq[0](x_in)
            y = self._apply_norm(seq[1], y, mask2d)
            y = seq[2](y)
            return y

        f1 = run_branch(self.b3, x)
        f2 = run_branch(self.b5, x)
        f3 = run_branch(self.bDil, x)

        if self.use_sk:
            y = self.sk((f1, f2, f3))
        else:
            y = (f1 + f2 + f3) / 3.0

        y = self.merge_pw(y)
        y = self._apply_norm(self.merge_gn, y, mask2d)
        y = self.act(y)
        y = self.drop(y)

        y = self.dw(y)
        y = self.pw(y)
        y = self._apply_norm(self.gn2, y, mask2d)
        y = self.act(y)

        y = self.res_scale(y) if self.use_res_scale else y
        if mask2d is not None:
            y = y * mask2d
        return z + y


# ============================================================
# cyc-axis conv1d (fixed large kernel)
# ============================================================

class CycAxisConv1d(nn.Module):
    """
    z: [B,C,cyc,pmax]
    u = mean over p -> [B,C,cyc]
    depthwise conv1d along cyc with fixed kernel, then gated, broadcast back to 2D.
    """
    def __init__(self, channels: int, kernel: int = 9, use_res_scale: bool = True):
        super().__init__()
        C = channels
        g = _best_gn_groups(C)
        k = int(kernel)
        k = max(3, k if k % 2 == 1 else k + 1)

        self.dw = nn.Conv1d(C, C, k, padding=k // 2, groups=C, bias=False)
        self.pw = nn.Conv1d(C, C, 1, bias=False)
        self.norm = nn.GroupNorm(g, C)
        self.act = nn.GELU()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(C, C, 1, bias=False), nn.Sigmoid())

        self.use_res_scale = bool(use_res_scale)
        self.res_scale = ResidualScale2D(C, init=1e-3) if self.use_res_scale else nn.Identity()

    def forward(self, z: torch.Tensor, mask2d: Optional[torch.Tensor] = None) -> torch.Tensor:
        u = z.mean(dim=-1)  # [B,C,cyc]
        x = self.dw(u)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        x = x * self.gate(u)
        x2d = x.unsqueeze(-1).expand_as(z)
        x2d = self.res_scale(x2d) if self.use_res_scale else x2d
        if mask2d is not None:
            x2d = x2d * mask2d
        return z + x2d


# ============================================================
# Non-periodic branch (Local-Global)
# ============================================================

class SimplifiedAttention(nn.Module):
    """Lightweight self-attention for [B,T,C]"""
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class LocalGlobalBranch(nn.Module):
    """
    x: [B,T,C] -> [B,T,C]
    local: dwconv3 + dwconvK
    global: pooled gate or simplified attention
    """
    def __init__(self, channels: int, local_kernel: int = 7, use_global_attn: bool = False):
        super().__init__()
        C = channels
        g = _best_gn_groups(C)
        k = int(local_kernel)
        k = max(3, k if k % 2 == 1 else k + 1)

        self.local3 = nn.Sequential(
            nn.Conv1d(C, C, 3, padding=1, groups=C, bias=False),
            nn.GroupNorm(g, C), nn.GELU()
        )
        self.localK = nn.Sequential(
            nn.Conv1d(C, C, k, padding=k // 2, groups=C, bias=False),
            nn.GroupNorm(g, C), nn.GELU()
        )
        self.merge = nn.Sequential(
            nn.Conv1d(C * 2, C, 1, bias=False),
            nn.GroupNorm(g, C), nn.GELU()
        )

        self.use_global_attn = bool(use_global_attn)
        if self.use_global_attn:
            self.attn = SimplifiedAttention(C, num_heads=1)
        else:
            hidden = max(8, C // 2)
            self.global_gate = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(C, hidden, 1, bias=False), nn.GELU(),
                nn.Conv1d(hidden, C, 1, bias=False), nn.Sigmoid()
            )

        self.res_scale = ResidualScale1D(C, init=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_nct = x.permute(0, 2, 1)  # [B,C,T]
        a = self.local3(x_nct)
        b = self.localK(x_nct)
        loc = self.merge(torch.cat([a, b], dim=1))  # [B,C,T]

        if self.use_global_attn:
            loc_ntc = loc.permute(0, 2, 1)
            glob_ntc = self.attn(loc_ntc)
            glob = glob_ntc.permute(0, 2, 1)
        else:
            gate = self.global_gate(loc)
            glob = loc * gate

        out = x_nct + self.res_scale(glob)
        return out.permute(0, 2, 1)  # [B,T,C]


# ============================================================
# Batched fold/unfold (unified 2D grid + mask)
# ============================================================

def fold_candidates_to_grid(
    s_flat: torch.Tensor,
    p_flat: torch.Tensor,
    T: int,
    p_max: int,
    cyc_max: int,
    reflect_pad: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    s_flat: [N,T,C]
    p_flat: [N]
    Returns:
      Z: [N,C,cyc_max,p_max]
      mask2d: [N,1,cyc_max,p_max] float {0,1}
      idx_map: (cyc_T, pos_T) each [N,T] for unfold
    """
    device = s_flat.device
    dtype = s_flat.dtype
    N, _, C = s_flat.shape

    p_flat = p_flat.clamp_min(1)
    tpad = ((T + p_flat - 1) // p_flat) * p_flat  # [N]
    tpad_max = int(tpad.max().item())

    t_all = torch.arange(tpad_max, device=device, dtype=torch.long).view(1, -1).expand(N, -1)  # [N,tpad_max]
    time_valid = t_all < tpad.view(N, 1)  # [N,tpad_max]

    if reflect_pad:
        src_idx = _reflect_indices(t_all, T)  # [N,tpad_max]
        s_src = s_flat.gather(1, src_idx.unsqueeze(-1).expand(N, tpad_max, C))  # [N,tpad_max,C]
    else:
        src_idx = t_all.clamp_max(T - 1)
        s_src = s_flat.gather(1, src_idx.unsqueeze(-1).expand(N, tpad_max, C))
        s_src = s_src * (t_all < T).unsqueeze(-1)

    s_src = s_src * time_valid.unsqueeze(-1).to(dtype)

    cyc_idx = t_all // p_flat.view(N, 1)
    pos_idx = t_all - cyc_idx * p_flat.view(N, 1)

    valid2d = time_valid & (pos_idx < p_flat.view(N, 1)) & (pos_idx < p_max) & (cyc_idx < cyc_max)

    HW = cyc_max * p_max
    spatial = (cyc_idx * p_max + pos_idx).clamp(0, HW - 1)  # [N,tpad_max]

    Z_flat = torch.zeros((N, C, HW), device=device, dtype=dtype)
    idx_sp = spatial.unsqueeze(1).expand(N, C, tpad_max)
    src = s_src.permute(0, 2, 1) * valid2d.unsqueeze(1).to(dtype)
    Z_flat.scatter_(2, idx_sp, src)
    Z = Z_flat.view(N, C, cyc_max, p_max)

    M_flat = torch.zeros((N, 1, HW), device=device, dtype=dtype)
    M_src = valid2d.to(dtype).unsqueeze(1)
    M_flat.scatter_(2, spatial.unsqueeze(1), M_src)
    mask2d = M_flat.view(N, 1, cyc_max, p_max)

    t_T = torch.arange(T, device=device, dtype=torch.long).view(1, -1).expand(N, -1)  # [N,T]
    cyc_T = (t_T // p_flat.view(N, 1)).clamp(0, cyc_max - 1)
    pos_T = (t_T - (t_T // p_flat.view(N, 1)) * p_flat.view(N, 1)).clamp(0, p_max - 1)

    return Z, mask2d, (cyc_T, pos_T)


def unfold_grid_to_series(
    Z: torch.Tensor,
    idx_map: Tuple[torch.Tensor, torch.Tensor],
    T: int
) -> torch.Tensor:
    """
    Z: [N,C,cyc_max,p_max]
    idx_map: (cyc_T, pos_T) each [N,T]
    Return y: [N,T,C]
    """
    N, C, cyc_max, p_max = Z.shape
    cyc_T, pos_T = idx_map
    HW = cyc_max * p_max
    Z_flat = Z.view(N, C, HW)
    spatial_T = (cyc_T * p_max + pos_T).clamp(0, HW - 1)
    idx_sp = spatial_T.unsqueeze(1).expand(N, C, T)
    y = Z_flat.gather(2, idx_sp)  # [N,C,T]
    return y.permute(0, 2, 1).contiguous()  # [N,T,C]


# ============================================================
# DSD Block: Periodic trunk + optional dual-path fusion
# ============================================================

class DSDBlock(nn.Module):
    def __init__(self, cfg: DSDNetConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.k = cfg.top_k
        self.min_p = cfg.min_period

        self.use_series_decomp = bool(cfg.use_series_decomp)
        self.decomp = SeriesDecomp(cfg.moving_avg) if self.use_series_decomp else None

        self.block2d = Inception2DResBlock(
            channels=cfg.d_model,
            use_sk=cfg.use_sk,
            sk_tau=cfg.sk_tau,
            use_res_scale=cfg.use_res_scale,
            drop=cfg.conv2d_dropout,
            use_masked_gn=cfg.use_masked_gn2d,
        )

        self.cyc1d = CycAxisConv1d(cfg.d_model, kernel=cfg.cyc_conv_kernel, use_res_scale=cfg.use_res_scale) \
            if cfg.use_cyc_conv1d else None

        self.gate_mlp = None
        if cfg.use_gate_mlp:
            hidden = max(8, cfg.d_model // 4)
            self.gate_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1)
            )

        self.local_branch = None
        if cfg.use_dual_path:
            self.local_branch = LocalGlobalBranch(
                channels=cfg.d_model,
                local_kernel=cfg.local_kernel,
                use_global_attn=cfg.use_global_attn
            )

            hidden = max(16, cfg.d_model // 4)
            self.gating_network = nn.Sequential(
                nn.Linear(cfg.d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, cfg.d_model)
            )
            nn.init.zeros_(self.gating_network[0].weight)
            nn.init.zeros_(self.gating_network[0].bias)
            nn.init.xavier_uniform_(self.gating_network[2].weight)
            nn.init.constant_(self.gating_network[2].bias, -2.0)

            self.register_buffer("_fw_steps", torch.zeros((), dtype=torch.long), persistent=False)

        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,C]
        """
        B, T, C = x.shape
        assert C == self.d_model

        if self.decomp is not None:
            s, t = self.decomp(x)  # [B,T,C], [B,T,C]
        else:
            s, t = x, torch.zeros_like(x)

        max_p = self.cfg.max_period if self.cfg.max_period is not None else (T // 2)
        idx, w_prior = auto_correlation_topk(s, self.k, min_p=self.min_p, max_p=max_p)  # [B,k], [B,k]

        N = B * self.k
        p_flat = idx.reshape(-1).to(torch.long)          # [N]
        w_flat = w_prior.reshape(-1).to(x.dtype)         # [N]

        s_flat = s.unsqueeze(1).expand(B, self.k, T, C).reshape(N, T, C).contiguous()

        p_max = int(max_p)
        cyc_max = int((T + self.min_p - 1) // self.min_p)  # ceil(T/min_p)

        Z, mask2d, idx_map = fold_candidates_to_grid(
            s_flat, p_flat=p_flat, T=T, p_max=p_max, cyc_max=cyc_max, reflect_pad=self.cfg.reflect_pad
        )
        Z = Z * mask2d

        Z = self.block2d(Z, mask2d=mask2d)
        if self.cyc1d is not None:
            Z = self.cyc1d(Z, mask2d=mask2d)

        Z = Z * mask2d
        Y = unfold_grid_to_series(Z, idx_map=idx_map, T=T)  # [N,T,C]

        w_term = (w_flat + self.eps).log().view(N, 1, 1)
        if self.gate_mlp is not None:
            gate = torch.sigmoid(self.gate_mlp(Y.mean(dim=1)))  # [N,1]
            g_term = (gate + self.eps).log().view(N, 1, 1)
            score = w_term + g_term
        else:
            score = w_term

        weight = torch.exp(score)  # [N,1,1]
        Yw = Y * weight

        Yw = Yw.view(B, self.k, T, C)
        weight = weight.view(B, self.k, 1, 1)

        agg_num = Yw.sum(dim=1)
        agg_den = weight.sum(dim=1).clamp_min(self.eps)
        agg = agg_num / agg_den

        periodic_out = agg + t + (x - t)  # == agg + x
        periodic_residual = periodic_out - x

        if not self.cfg.use_dual_path or self.local_branch is None:
            return periodic_out

        local_out = self.local_branch(x)
        local_res = local_out - x

        if self.cfg.use_orthogonal:
            local_res = _orth_residual(local_res, periodic_residual.detach())

        if self.cfg.use_spa_gate:
            with torch.no_grad():
                conf = _period_confidence_from_wprior(w_prior)  # [B]
        else:
            conf = torch.zeros((B,), device=x.device, dtype=x.dtype)

        x_global = x.mean(dim=1)
        g_logit = self.gating_network(x_global)

        if self.cfg.use_spa_gate:
            g_logit = g_logit - self.cfg.gate_alpha * (conf - self.cfg.gate_tau).unsqueeze(-1)

        gate = torch.sigmoid(g_logit).unsqueeze(1)  # [B,1,C]

        # warmup: 近似（建议更严格的 warmup 放到训练 loop）
        if self.training and self.cfg.gate_warmup_epochs > 0:
            self._fw_steps = self._fw_steps + 1
            if self._fw_steps.item() < int(self.cfg.gate_warmup_epochs * 400):
                gate = gate * 0.0

        out = periodic_out + gate * (self.cfg.dual_beta * local_res)
        return out


# ============================================================
# DSD-Net model
# ============================================================

class DSDNet(nn.Module):
    """
    Integrated DSD-Net: project -> stacked DSDBlock -> norm -> pooling -> classifier
    """
    def __init__(self, cfg: DSDNetConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model

        self.project = nn.Linear(cfg.enc_in, cfg.d_model)

        self.blocks = nn.ModuleList([DSDBlock(cfg) for _ in range(cfg.e_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.e_layers)])

        self.use_attention_pool = bool(cfg.use_attention_pool)
        if self.use_attention_pool:
            self.pool_q = nn.Parameter(torch.randn(1, 1, cfg.d_model))

        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_class)

        self.logit_T = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def attention_pool(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.pool_q.expand(B, -1, -1)
        att = (q @ x.transpose(1, 2)) / (C ** 0.5)
        att = torch.softmax(att, dim=-1)
        return (att @ x).squeeze(1)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        x_enc: [B,T,enc_in]
        return logits: [B,num_class]
        """
        x = self.project(x_enc)
        for blk, ln in zip(self.blocks, self.norms):
            x = ln(blk(x))

        if self.use_attention_pool:
            x = self.attention_pool(x)
        else:
            x = x.mean(dim=1)

        x = self.dropout(x)
        logits = self.classifier(x) / self.logit_T.clamp_min(1e-6)
        return logits


# ============================================================
# TimesNet repo wrapper: must expose class Model
# ============================================================

class Model(nn.Module):
    """
    TimesNet 仓库兼容包装层：
    Exp_Classification 会调用：Model(args).forward(batch_x, padding_mask, None, None)
    """
    def __init__(self, args):
        super().__init__()
        cfg = build_cfg_from_args(args)
        self.net = DSDNet(cfg)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, **kwargs):
        # classification: x_mark_enc 通常是 padding_mask；当前实现不依赖它
        return self.net(x_enc)
