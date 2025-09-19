import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# ===== 继承你的 MyModule =====
from models import MyModule

# ===== 基础工具 =====
def _best_gn_groups(C: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if C % g == 0:
            return g
    return 1

class ResidualScale(nn.Module):
    def __init__(self, channels: int, init: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * init)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                     padding=(kernel_size - 1) // 2, count_include_pad=False)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_nct = x.permute(0, 2, 1)
        trend = self.avg_pool(x_nct).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend

def auto_correlation(x: torch.Tensor, k: int, min_p: int = 3, max_p: int = None):
    B, T, C = x.shape
    dtype = x.dtype
    if max_p is None:
        max_p = max(1, T // 2)
    max_p = max(min_p, min(max_p, T - 1))
    Xf = torch.fft.rfft(x, dim=1)
    P  = (Xf * torch.conj(Xf)).real
    Pm = P.mean(dim=-1)
    r  = torch.fft.irfft(Pm, n=T, dim=1)

    mask = torch.ones_like(r, dtype=torch.bool)
    mask[:, 0] = False
    if min_p > 1: mask[:, 1:min_p] = False
    if max_p + 1 < T: mask[:, max_p + 1:] = False

    very_neg = torch.finfo(dtype).min / 8 if dtype.is_floating_point else -1e9
    r_masked = torch.where(mask, r, torch.full_like(r, very_neg))

    # p0 = max(min_p, min(max_p, max(2, T // 4)))
    # if (mask.sum(dim=1) == 0).any():
    #     r_masked[:] = very_neg
    #     r_masked[:, p0] = 0.0
    # 逐样本回退
    p0 = max(min_p, min(max_p, max(2, T // 4)))
    all_bad = (mask.sum(dim=1) == 0)
    if all_bad.any():
        r_masked[all_bad] = very_neg
        r_masked[all_bad, p0] = 0.0

    k_eff = min(k, max_p - min_p + 1)
    vals, idx = torch.topk(r_masked, k=k_eff, dim=1)
    if k_eff < k:
        pad_n = k - k_eff
        idx  = torch.cat([idx,  idx[:, -1:].expand(B, pad_n)], dim=1)
        vals = torch.cat([vals, vals[:, -1:].expand(B, pad_n)], dim=1)
    w = F.softmax(vals, dim=1)
    return idx, w

def _fold_2d(x_1d: torch.Tensor, p: int, reflect_pad: bool) -> Tuple[torch.Tensor, int]:
    """x_1d:[b,T,C] -> z:[b,C,cyc,p]"""
    b, T, C = x_1d.shape
    pad = ((T + p - 1) // p) * p - T
    if pad > 0:
        x_nct = x_1d.permute(0, 2, 1).contiguous()
        mode = 'reflect' if reflect_pad else 'constant'
        x_nct = F.pad(x_nct, (0, pad), mode=mode, value=0.0 if not reflect_pad else 0.0)
        x_1d  = x_nct.permute(0, 2, 1).contiguous()
    T_new = x_1d.shape[1]
    cyc = T_new // p
    z = x_1d.reshape(b, cyc, p, C).permute(0, 3, 1, 2).contiguous()
    return z, T

def _unfold_1d(z_2d: torch.Tensor, T: int) -> torch.Tensor:
    """z_2d:[b,C,cyc,p] -> y:[b,T,C]"""
    b, C, cyc, p = z_2d.shape
    y = z_2d.permute(0, 2, 3, 1).reshape(b, cyc * p, C)[:, :T, :].contiguous()
    return y

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(reduction, channels // 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False), nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False), nn.Sigmoid()
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.fc(z)

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
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.tau = tau
    def forward(self, feats: List[torch.Tensor], w_prior: torch.Tensor = None):
        B, C, _, _ = feats[0].shape
        s = sum(feats) / self.n
        a = self.squeeze(s).view(B, self.n, C, 1, 1)
        if w_prior is not None:
            if w_prior.dim() == 4:      # [B,1,1,1]
                w_prior = w_prior.view(B, 1, 1, 1, 1).expand(B, self.n, C, 1, 1)
            elif w_prior.dim() == 5 and w_prior.size(1) == 1:  # [B,1,1,1,1]
                w_prior = w_prior.expand(B, self.n, C, 1, 1)
            a = a + self.alpha * w_prior
        w = torch.softmax(a / self.tau, dim=1)
        out = sum(w[:, i] * feats[i] for i in range(self.n))
        return out

# ===== 带开关的 Inception2D 残差块 =====
class Inception2dResBlockAdv(nn.Module):
    def __init__(self, channels: int, use_sk: bool, use_se: bool, se_strength: float,
                 use_res_scale: bool, sk_tau: float, drop: float = 0.0):
        super().__init__()
        C = channels; g = _best_gn_groups(C)
        self.pre = nn.GroupNorm(g, C)

        self.b3   = nn.Sequential(nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
                                  nn.GroupNorm(g, C), nn.GELU())
        self.b5   = nn.Sequential(nn.Conv2d(C, C, 5, padding=2, groups=C, bias=False),
                                  nn.GroupNorm(g, C), nn.GELU())
        self.bDil = nn.Sequential(nn.Conv2d(C, C, 3, padding=2, dilation=2, groups=C, bias=False),
                                  nn.GroupNorm(g, C), nn.GELU())

        self.use_sk = bool(use_sk)
        self.sk     = SKMerge(C, n_branches=3, reduction=max(8, C // 4), tau=sk_tau) if self.use_sk else None

        self.merge_pw = nn.Conv2d(C, C, 1, bias=False)
        self.merge_gn = nn.GroupNorm(g, C)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.dw  = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.pw  = nn.Conv2d(C, C, 1, bias=False)
        self.gn2 = nn.GroupNorm(g, C)

        self.use_se = bool(use_se)
        self.se_strength = float(se_strength)
        self.se = SEBlock(C, reduction=max(8, C // 4)) if self.use_se else None

        self.use_res_scale = bool(use_res_scale)
        self.res_scale = ResidualScale(C, init=1e-3) if self.use_res_scale else nn.Identity()

    def forward(self, z: torch.Tensor, w_prior_branch: torch.Tensor = None) -> torch.Tensor:
        x_in = self.pre(z)
        f1, f2, f3 = self.b3(x_in), self.b5(x_in), self.bDil(x_in)
        if self.use_sk:
            x = self.sk([f1, f2, f3], w_prior=w_prior_branch)
        else:
            x = (f1 + f2 + f3) / 3.0

        x = self.merge_pw(x); x = self.merge_gn(x); x = self.act(x); x = self.drop(x)
        x = self.dw(x); x = self.pw(x); x = self.gn2(x); x = self.act(x)

        if self.use_se and self.se_strength > 0:
            x = x * (self.se(x).pow(self.se_strength))

        x = self.res_scale(x) if self.use_res_scale else x
        return z + x

# ===== 带开关的 cyc 轴大核 conv1d =====
class CycLargeKernelConv1dAdv(nn.Module):
    def __init__(self, channels: int, k: int, use_res_scale: bool):
        super().__init__()
        C = channels; g = _best_gn_groups(C)
        self.k_cfg = int(k)
        self.dw3 = nn.Conv1d(C, C, 3, padding=1, groups=C, bias=False)
        self.pw  = nn.Conv1d(C, C, 1, bias=False)
        self.norm = nn.GroupNorm(g, C)
        self.act  = nn.GELU()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(C, C, 1, bias=False), nn.Sigmoid())
        self.use_res_scale = bool(use_res_scale)
        self.res_scale = ResidualScale(C, init=1e-3) if self.use_res_scale else nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        u = z.mean(dim=-1)  # [B,C,cyc]
        cyc = u.size(-1)
        if cyc <= 1:
            return z
        k_eff = max(3, min(self.k_cfg, 2 * cyc - 1))
        rep = max(1, (k_eff // 3))
        x = u
        for _ in range(rep):
            x = self.dw3(x)
        x = self.pw(x); x = self.norm(x); x = self.act(x)
        g = self.gate(u); x = x * g
        x = x.unsqueeze(-1).expand_as(z)
        x = self.res_scale(x) if self.use_res_scale else x
        return z + x

# ===== 带开关的 TimesBlock =====
class AdvTimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 基本超参
        self.k        = configs.top_k
        self.kernel   = configs.moving_avg
        self.d_model  = configs.d_model
        self.min_p    = getattr(configs, 'min_period', 3)
        self.drop2d   = getattr(configs, 'conv2d_dropout', 0.0)

        # 开关
        self.use_series_decomp = bool(getattr(configs, 'use_series_decomp', 1))
        self.use_sk            = bool(getattr(configs, 'use_sk', 1))
        self.use_se            = bool(getattr(configs, 'use_se', 0))
        self.se_strength       = float(getattr(configs, 'se_strength', 0.0))
        self.use_cyc_conv1d    = bool(getattr(configs, 'use_cyc_conv1d', 1))
        self.cyc_k             = int(getattr(configs, 'cyc_conv_kernel', 9))
        self.use_gate_mlp      = bool(getattr(configs, 'use_gate_mlp', 1))
        self.use_res_scale     = bool(getattr(configs, 'use_res_scale', 1))
        self.reflect_pad       = bool(getattr(configs, 'reflect_pad', 1))
        self.sk_tau            = float(getattr(configs, 'sk_tau', 1.5))

        # 组件
        self.decomp = SeriesDecomp(self.kernel) if self.use_series_decomp else None
        self.block2d = Inception2dResBlockAdv(
            self.d_model, use_sk=self.use_sk, use_se=self.use_se,
            se_strength=self.se_strength, use_res_scale=self.use_res_scale,
            sk_tau=self.sk_tau, drop=self.drop2d
        )
        self.cyc1d = (CycLargeKernelConv1dAdv(self.d_model, k=self.cyc_k, use_res_scale=self.use_res_scale)
                      if self.use_cyc_conv1d else None)

        hidden = max(8, self.d_model // 4)
        self.gate_mlp = (nn.Sequential(nn.Linear(self.d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))
                         if self.use_gate_mlp else None)
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,C=d_model]
        """
        B, T, C = x.shape
        assert C == self.d_model, "Input last dim must equal d_model"

        # 分解（可关）
        if self.decomp is not None:
            s, t = self.decomp(x)
        else:
            s, t = x, 0.0

        # 周期候选
        idx, w_prior = auto_correlation(s, self.k, min_p=self.min_p, max_p=T // 2)  # [B,k],[B,k]

        agg_num = torch.zeros_like(x)
        agg_den = x.new_zeros(B)

        # 以 unique(p) 聚合，减少重复折叠
        unique_p = torch.unique(idx)
        for pv in unique_p.tolist():
            mask = (idx == pv)
            if not mask.any():
                continue
            b_idx, j_idx = mask.nonzero(as_tuple=True)
            m = b_idx.numel()

            sb = s[b_idx]                                # [m,T,C]
            wb = w_prior[b_idx, j_idx].view(m, 1, 1)     # [m,1,1]

            # 折叠到 2D（可选反射填充）
            z, _T = _fold_2d(sb, int(pv), reflect_pad=self.reflect_pad)  # [m,C,cyc,p]

            # 多尺度 2D +（可选 SK/SE/残差缩放）
            z = self.block2d(z)                          # [m,C,cyc,p]

            # 可选：cyc 轴大核卷积
            if self.cyc1d is not None:
                z = self.cyc1d(z)

            # 展回 1D
            y = _unfold_1d(z, _T)                        # [m,T,C]

            # 融合权重：w_prior ×（可选 gate）
            if self.gate_mlp is not None:
                gate = torch.sigmoid(self.gate_mlp(y.mean(dim=1)))  # [m,1]
                score = torch.log(wb + self.eps) + torch.log(gate.view(m, 1, 1) + self.eps)
            else:
                score = torch.log(wb + self.eps)
            score_exp = torch.exp(score)                 # [m,1,1]

            contrib = y * score_exp                      # [m,T,C]
            agg_num.index_add_(0, b_idx, contrib)
            agg_den.index_add_(0, b_idx, score_exp.view(-1))

        agg = agg_num / (agg_den.view(B, 1, 1) + self.eps)

        # 加回趋势，并保留季节残差（若没分解则 t=0）
        out = agg + (t if isinstance(t, torch.Tensor) else 0.0)
        return out + (x - (t if isinstance(t, torch.Tensor) else 0.0))

# ===== 继承 MyModule.Model，并替换 blocks =====
class Model(MyModule.Model):
    """
    继承你的 MyModule.Model：
    - super().__init__(configs) 后，直接替换 self.blocks 为 AdvTimesBlock 的 ModuleList
    - 其它（project / norms / attention_pool / classifier）完全沿用父类实现
    - 因为 forward 会遍历 self.blocks，所以无需重写 forward
    """
    def __init__(self, configs):
        super().__init__(configs)
        # 用带开关的 AdvTimesBlock 替换父类构建的 blocks
        try:
            self.blocks = nn.ModuleList([AdvTimesBlock(configs) for _ in range(configs.e_layers)])
        except Exception as e:
            print(f"[TimesNetAdv] WARNING: fallback to MyModule blocks due to: {e}")
            # 出现异常则保留父类 blocks，不影响正常训练/推理
            pass
