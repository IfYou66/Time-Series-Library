# =================== TimesNetAdv（主干+门控旁路，修复版） ===================
import math
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
    """用于 4D 张量（如 [B,C,H,W] / [B,C,cyc,p]）的残差缩放"""
    def __init__(self, channels: int, init: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class ResidualScale1D(nn.Module):
    """用于 3D 张量 [B,C,T] 的残差缩放"""
    def __init__(self, channels: int, init: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1) * init)

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
    P = (Xf * torch.conj(Xf)).real
    Pm = P.mean(dim=-1)
    r = torch.fft.irfft(Pm, n=T, dim=1)

    mask = torch.ones_like(r, dtype=torch.bool)
    mask[:, 0] = False
    if min_p > 1:
        mask[:, 1:min_p] = False
    if max_p + 1 < T:
        mask[:, max_p + 1:] = False

    very_neg = torch.finfo(dtype).min / 8 if dtype.is_floating_point else -1e9
    r_masked = torch.where(mask, r, torch.full_like(r, very_neg))

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
    return idx, w  # [B,k], [B,k]


def _fold_2d(x_1d: torch.Tensor, p: int, reflect_pad: bool) -> Tuple[torch.Tensor, int]:
    """x_1d:[b,T,C] -> z:[b,C,cyc,p]"""
    b, T, C = x_1d.shape
    pad = ((T + p - 1) // p) * p - T
    if pad > 0:
        x_nct = x_1d.permute(0, 2, 1).contiguous()
        mode = 'reflect' if reflect_pad else 'constant'
        x_nct = F.pad(x_nct, (0, pad), mode=mode, value=0.0 if not reflect_pad else 0.0)
        x_1d = x_nct.permute(0, 2, 1).contiguous()
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
        C = channels
        g = _best_gn_groups(C)
        self.pre = nn.GroupNorm(g, C)

        self.b3 = nn.Sequential(nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
                                nn.GroupNorm(g, C), nn.GELU())
        self.b5 = nn.Sequential(nn.Conv2d(C, C, 5, padding=2, groups=C, bias=False),
                                nn.GroupNorm(g, C), nn.GELU())
        self.bDil = nn.Sequential(nn.Conv2d(C, C, 3, padding=2, dilation=2, groups=C, bias=False),
                                  nn.GroupNorm(g, C), nn.GELU())

        self.use_sk = bool(use_sk)
        self.sk = SKMerge(C, n_branches=3, reduction=max(8, C // 4), tau=sk_tau) if self.use_sk else None

        self.merge_pw = nn.Conv2d(C, C, 1, bias=False)
        self.merge_gn = nn.GroupNorm(g, C)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.dw = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.pw = nn.Conv2d(C, C, 1, bias=False)
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

        x = self.merge_pw(x)
        x = self.merge_gn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.gn2(x)
        x = self.act(x)

        if self.use_se and self.se_strength > 0:
            x = x * (self.se(x).pow(self.se_strength))

        x = self.res_scale(x) if self.use_res_scale else x
        return z + x


# ===== 带开关的 cyc 轴大核 conv1d =====
class CycLargeKernelConv1dAdv(nn.Module):
    def __init__(self, channels: int, k: int, use_res_scale: bool):
        super().__init__()
        C = channels
        g = _best_gn_groups(C)
        self.k_cfg = int(k)
        self.dw3 = nn.Conv1d(C, C, 3, padding=1, groups=C, bias=False)
        self.pw = nn.Conv1d(C, C, 1, bias=False)
        self.norm = nn.GroupNorm(g, C)
        self.act = nn.GELU()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(C, C, 1, bias=False), nn.Sigmoid())
        self.use_res_scale = bool(use_res_scale)
        self.res_scale = ResidualScale(C, init=1e-3) if self.use_res_scale else nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,C,cyc,p]
        u = z.mean(dim=-1)  # [B,C,cyc]
        cyc = u.size(-1)
        if cyc <= 1:
            return z
        k_eff = max(3, min(self.k_cfg, 2 * cyc - 1))
        rep = max(1, (k_eff // 3))
        x = u
        for _ in range(rep):
            x = self.dw3(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        g = self.gate(u)
        x = x * g
        x = x.unsqueeze(-1).expand_as(z)  # [B,C,cyc,p]
        x = self.res_scale(x) if self.use_res_scale else x
        return z + x


# ===== 新增：局部-全局分支（非周期路径）=====
class LocalGlobalBranch(nn.Module):
    """
    不依赖周期假设的特征提取分支
    - 局部：深度可分离卷积捕获短程模式
    - 全局：简化自注意力或全局平均
    """

    def __init__(self, channels: int, local_kernel: int = 7, use_global_attn: bool = False):
        super().__init__()
        C = channels
        g = _best_gn_groups(C)

        # 局部路径：多尺度深度可分离卷积
        self.local_conv3 = nn.Sequential(
            nn.Conv1d(C, C, 3, padding=1, groups=C, bias=False),
            nn.GroupNorm(g, C), nn.GELU()
        )
        self.local_conv_large = nn.Sequential(
            nn.Conv1d(C, C, local_kernel, padding=local_kernel // 2, groups=C, bias=False),
            nn.GroupNorm(g, C), nn.GELU()
        )

        # 融合局部特征
        self.local_merge = nn.Sequential(
            nn.Conv1d(C * 2, C, 1, bias=False),
            nn.GroupNorm(g, C), nn.GELU()
        )

        # 全局路径
        self.use_global_attn = use_global_attn
        if use_global_attn:
            self.global_attn = SimplifiedAttention(C, num_heads=1)
        else:
            hidden = max(8, C // 2)
            self.global_pool = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(C, hidden, 1, bias=False), nn.GELU(),
                nn.Conv1d(hidden, C, 1, bias=False), nn.Sigmoid()
            )

        # 残差缩放（针对 3D 特征）
        self.res_scale = ResidualScale1D(C, init=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,C] -> [B,T,C]
        """
        x_nct = x.permute(0, 2, 1)  # [B,C,T]

        # 局部特征提取
        local3 = self.local_conv3(x_nct)
        local_large = self.local_conv_large(x_nct)
        local_feat = self.local_merge(torch.cat([local3, local_large], dim=1))  # [B,C,T]

        # 全局特征提取
        if self.use_global_attn:
            local_feat_ntc = local_feat.permute(0, 2, 1)  # [B,T,C]
            global_feat = self.global_attn(local_feat_ntc)  # [B,T,C]
            global_feat = global_feat.permute(0, 2, 1)  # [B,C,T]
        else:
            gate = self.global_pool(local_feat)  # [B,C,1]
            global_feat = local_feat * gate  # [B,C,T]

        # 残差连接（保持 3D）
        out = x_nct + self.res_scale(global_feat)  # [B,C,T]
        return out.permute(0, 2, 1)  # [B,T,C]


class SimplifiedAttention(nn.Module):
    """简化的单头自注意力（降低计算量）"""

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels

        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,T,C]"""
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, T, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)  # [B,T,C]
        out = self.proj(out)
        return out


# ====== 工具：频域置信度与正交残差（修复版） ======
def _period_confidence_from_wprior(w_prior: torch.Tensor, eps: float = 1e-8):
    """
    w_prior: [B, k]（自相关 top-k 的 softmax 权重）
    返回 conf∈[0,1]，越接近1表示周期越明确
    """
    H = -(w_prior * (w_prior + eps).log()).sum(dim=1)    # [B]
    k = w_prior.size(1)
    if k <= 1:
        return torch.zeros_like(H)
    H_max = torch.log(torch.tensor(k, dtype=w_prior.dtype, device=w_prior.device) + eps)
    conf = 1.0 - (H / (H_max + eps))
    return conf.clamp(0.0, 1.0)  # [B]


def _orth_residual(local_res: torch.Tensor, periodic_res: torch.Tensor, eps=1e-6):
    """
    对时间维做投影消除：local - proj(local on periodic)
    形状均为 [B,T,C]
    """
    num = (local_res * periodic_res).sum(dim=1, keepdim=True)            # [B,1,C]
    den = (periodic_res * periodic_res).sum(dim=1, keepdim=True).add(eps)
    proj = (num / den) * periodic_res                                   # [B,T,C]
    return local_res - proj


# ===== 带双路径的 TimesBlock（主干+门控旁路） =====
class AdvTimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 基本超参
        self.k = configs.top_k
        self.kernel = configs.moving_avg
        self.d_model = configs.d_model
        self.min_p = getattr(configs, 'min_period', 3)
        self.drop2d = getattr(configs, 'conv2d_dropout', 0.0)

        # 开关
        self.use_series_decomp = bool(getattr(configs, 'use_series_decomp', 1))
        self.use_sk = bool(getattr(configs, 'use_sk', 1))
        self.use_se = bool(getattr(configs, 'use_se', 0))
        self.se_strength = float(getattr(configs, 'se_strength', 0.0))
        self.use_cyc_conv1d = bool(getattr(configs, 'use_cyc_conv1d', 1))
        self.cyc_k = int(getattr(configs, 'cyc_conv_kernel', 9))
        self.use_gate_mlp = bool(getattr(configs, 'use_gate_mlp', 1))
        self.use_res_scale = bool(getattr(configs, 'use_res_scale', 1))
        self.reflect_pad = bool(getattr(configs, 'reflect_pad', 1))
        self.sk_tau = float(getattr(configs, 'sk_tau', 1.5))

        # 双路径
        self.use_dual_path = bool(getattr(configs, 'use_dual_path', 0))
        self.local_kernel = int(getattr(configs, 'local_kernel', 7))
        self.use_global_attn = bool(getattr(configs, 'use_global_attn', 0))

        # 组件
        self.decomp = SeriesDecomp(self.kernel) if self.use_series_decomp else None

        # 周期主干
        self.block2d = Inception2dResBlockAdv(
            self.d_model, use_sk=self.use_sk, use_se=self.use_se,
            se_strength=self.se_strength, use_res_scale=self.use_res_scale,
            sk_tau=self.sk_tau, drop=self.drop2d
        )
        self.cyc1d = (CycLargeKernelConv1dAdv(self.d_model, k=self.cyc_k, use_res_scale=self.use_res_scale)
                      if self.use_cyc_conv1d else None)

        # 局部-全局旁路
        self.local_global_branch = (LocalGlobalBranch(self.d_model,
                                                      local_kernel=self.local_kernel,
                                                      use_global_attn=self.use_global_attn)
                                    if self.use_dual_path else None)

        # ===== 动态门控（per-channel），默认偏闭；支持频域先验 + warmup =====
        if self.use_dual_path:
            hidden = max(16, self.d_model // 4)
            self.gating_network = nn.Sequential(
                nn.Linear(self.d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.d_model)
            )
            # 初始化：让门默认偏闭（对 Hell 友好）
            nn.init.zeros_(self.gating_network[0].weight)
            nn.init.zeros_(self.gating_network[0].bias)
            nn.init.xavier_uniform_(self.gating_network[2].weight)
            nn.init.constant_(self.gating_network[2].bias, -2.0)

            # 频域先验强度/阈值 + 旁路幅度系数 + 可选 warmup（按 epoch 粗估）
            self.gate_alpha = float(getattr(configs, 'gate_alpha', 4.0))
            self.gate_tau   = float(getattr(configs, 'gate_tau',   0.5))
            self.dual_beta  = float(getattr(configs, 'dual_beta',  0.3))
            self.gate_warmup = int(getattr(configs, 'gate_warmup', 0))  # 0 表示不启用 warmup
            self._fw_steps = 0  # forward 步数计数器（warmup 用）

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

        # ===== 周期主干 =====
        if self.decomp is not None:
            s, t = self.decomp(x)
        else:
            s, t = x, 0.0

        idx, w_prior = auto_correlation(s, self.k, min_p=self.min_p, max_p=T // 2)  # [B,k],[B,k]

        agg_num = torch.zeros_like(x)
        agg_den = x.new_zeros(B)

        unique_p = torch.unique(idx)
        for pv in unique_p.tolist():
            mask = (idx == pv)
            if not mask.any():
                continue
            b_idx, j_idx = mask.nonzero(as_tuple=True)
            m = b_idx.numel()

            sb = s[b_idx]  # [m,T,C]
            wb = w_prior[b_idx, j_idx].view(m, 1, 1)  # [m,1,1]

            z, _T = _fold_2d(sb, int(pv), reflect_pad=self.reflect_pad)  # [m,C,cyc,p]
            z = self.block2d(z)  # [m,C,cyc,p]
            if self.cyc1d is not None:
                z = self.cyc1d(z)
            y = _unfold_1d(z, _T)  # [m,T,C]

            if self.gate_mlp is not None:
                gate = torch.sigmoid(self.gate_mlp(y.mean(dim=1)))  # [m,1]
                score = torch.log(wb + self.eps) + torch.log(gate.view(m, 1, 1) + self.eps)
            else:
                score = torch.log(wb + self.eps)
            score_exp = torch.exp(score)  # [m,1,1]

            contrib = y * score_exp  # [m,T,C]
            agg_num.index_add_(0, b_idx, contrib)
            agg_den.index_add_(0, b_idx, score_exp.view(-1))

        agg = agg_num / (agg_den.view(B, 1, 1) + self.eps)

        # 主干输出：x + periodic_residual
        periodic_out = agg + (t if isinstance(t, torch.Tensor) else 0.0)
        periodic_out = periodic_out + (x - (t if isinstance(t, torch.Tensor) else 0.0))
        periodic_residual = periodic_out - x  # [B,T,C]

        # ===== 旁路：Local-Global（可选）=====
        if self.use_dual_path and self.local_global_branch is not None:
            local_global_out = self.local_global_branch(x)      # [B,T,C]
            local_residual = local_global_out - x               # [B,T,C]

            # 正交化：只保留互补信息（不反向影响主干）
            local_residual = _orth_residual(local_residual, periodic_residual.detach())

            # 频域先验（无梯度，周期越强门越关）
            with torch.no_grad():
                conf = _period_confidence_from_wprior(w_prior)  # [B]

            # learned gate（per-channel）+ 正确的“减号”先验偏置
            x_global = x.mean(dim=1)                            # [B,C]
            g_logit = self.gating_network(x_global)             # [B,C]
            g_logit = g_logit - self.gate_alpha * (conf - self.gate_tau).unsqueeze(-1)
            gate = torch.sigmoid(g_logit).unsqueeze(1)          # [B,1,C]

            # 可选：warmup 冻结旁路前若干 epoch（粗估 steps_per_epoch=374，来自你的 Hell 日志）
            if self.training and self.gate_warmup > 0:
                self._fw_steps += 1
                steps_per_epoch = 374
                if self._fw_steps < steps_per_epoch * self.gate_warmup:
                    gate = gate * 0.0

            # 限幅叠加：防过补
            out = periodic_out + gate * (self.dual_beta * local_residual)
            return out

        # 仅主干
        return periodic_out


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
        try:
            self.blocks = nn.ModuleList([AdvTimesBlock(configs) for _ in range(configs.e_layers)])
            print(f"[TimesNetAdv] Successfully initialized with dual_path={getattr(configs, 'use_dual_path', 0)}")
        except Exception as e:
            print(f"[TimesNetAdv] WARNING: fallback to MyModule blocks due to: {e}")
            pass



# ------------------------------
# 顶层 Model（继承你的 MyModule.Model）
# ------------------------------
class Model(MyModule.Model):
    """
    - super().__init__(configs) 后，替换 self.blocks 为 AdvTimesBlock 列表
    - 暴露 logit_T（训练脚本用于温度/校准）
    - pop_block_aux_losses(): 收集 blocks 的辅助损失（自相关熵等）
    """
    def __init__(self, configs):
        super().__init__(configs)
        try:
            self.blocks = nn.ModuleList([AdvTimesBlock(configs) for _ in range(configs.e_layers)])
            print(f"[TimesNetAdv] Successfully initialized with dual_path={getattr(configs, 'use_dual_path', 0)}")
        except Exception as e:
            print(f"[TimesNetAdv] WARNING: fallback to MyModule blocks due to: {e}")
            pass

        # 分类温度（训练用可学习，评测可覆盖）
        self.logit_T = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # 正则权重（从 configs 读取，供训练脚本参考）
        self.lambda_sparse = float(getattr(configs, 'lambda_sparse', 0.0))
        self.is_hell = bool(getattr(configs, 'is_hell', False))

    def pop_block_aux_losses(self):
        aux = {}
        if hasattr(self, 'blocks'):
            for b in self.blocks:
                if hasattr(b, 'pop_aux_losses'):
                    one = b.pop_aux_losses()
                    for k, v in one.items():
                        aux[k] = (aux.get(k, 0.0) + v)
        return aux


# =================== TimesNetAdv（主干+门控旁路，修复版） ===================
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, List
#
# # ===== 继承你的 MyModule =====
# from models import MyModule
#
#
# # ===== 基础工具 =====
# def _best_gn_groups(C: int) -> int:
#     for g in [32, 16, 8, 4, 2, 1]:
#         if C % g == 0:
#             return g
#     return 1
#
#
# class ResidualScale(nn.Module):
#     """用于 4D 张量（如 [B,C,H,W] / [B,C,cyc,p]）的残差缩放"""
#     def __init__(self, channels: int, init: float = 1e-3):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * init)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * self.gamma
#
#
# class ResidualScale1D(nn.Module):
#     """用于 3D 张量 [B,C,T] 的残差缩放"""
#     def __init__(self, channels: int, init: float = 1e-3):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(1, channels, 1) * init)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * self.gamma
#
#
# class SeriesDecomp(nn.Module):
#     def __init__(self, kernel_size: int):
#         super().__init__()
#         self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
#                                      padding=(kernel_size - 1) // 2, count_include_pad=False)
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x_nct = x.permute(0, 2, 1)
#         trend = self.avg_pool(x_nct).permute(0, 2, 1)
#         seasonal = x - trend
#         return seasonal, trend
#
#
# def auto_correlation(x: torch.Tensor, k: int, min_p: int = 3, max_p: int = None):
#     B, T, C = x.shape
#     dtype = x.dtype
#     if max_p is None:
#         max_p = max(1, T // 2)
#     max_p = max(min_p, min(max_p, T - 1))
#     Xf = torch.fft.rfft(x, dim=1)
#     P = (Xf * torch.conj(Xf)).real
#     Pm = P.mean(dim=-1)
#     r = torch.fft.irfft(Pm, n=T, dim=1)
#
#     mask = torch.ones_like(r, dtype=torch.bool)
#     mask[:, 0] = False
#     if min_p > 1:
#         mask[:, 1:min_p] = False
#     if max_p + 1 < T:
#         mask[:, max_p + 1:] = False
#
#     very_neg = torch.finfo(dtype).min / 8 if dtype.is_floating_point else -1e9
#     r_masked = torch.where(mask, r, torch.full_like(r, very_neg))
#
#     p0 = max(min_p, min(max_p, max(2, T // 4)))
#     all_bad = (mask.sum(dim=1) == 0)
#     if all_bad.any():
#         r_masked[all_bad] = very_neg
#         r_masked[all_bad, p0] = 0.0
#
#     k_eff = min(k, max_p - min_p + 1)
#     vals, idx = torch.topk(r_masked, k=k_eff, dim=1)
#     if k_eff < k:
#         pad_n = k - k_eff
#         idx = torch.cat([idx, idx[:, -1:].expand(B, pad_n)], dim=1)
#         vals = torch.cat([vals, vals[:, -1:].expand(B, pad_n)], dim=1)
#     w = F.softmax(vals, dim=1)
#     return idx, w  # [B,k], [B,k]
#
#
# def _fold_2d(x_1d: torch.Tensor, p: int, reflect_pad: bool) -> Tuple[torch.Tensor, int]:
#     """x_1d:[b,T,C] -> z:[b,C,cyc,p]"""
#     b, T, C = x_1d.shape
#     pad = ((T + p - 1) // p) * p - T
#     if pad > 0:
#         x_nct = x_1d.permute(0, 2, 1).contiguous()
#         mode = 'reflect' if reflect_pad else 'constant'
#         x_nct = F.pad(x_nct, (0, pad), mode=mode, value=0.0 if not reflect_pad else 0.0)
#         x_1d = x_nct.permute(0, 2, 1).contiguous()
#     T_new = x_1d.shape[1]
#     cyc = T_new // p
#     z = x_1d.reshape(b, cyc, p, C).permute(0, 3, 1, 2).contiguous()
#     return z, T
#
#
# def _unfold_1d(z_2d: torch.Tensor, T: int) -> torch.Tensor:
#     """z_2d:[b,C,cyc,p] -> y:[b,T,C]"""
#     b, C, cyc, p = z_2d.shape
#     y = z_2d.permute(0, 2, 3, 1).reshape(b, cyc * p, C)[:, :T, :].contiguous()
#     return y
#
#
# class SEBlock(nn.Module):
#     def __init__(self, channels: int, reduction: int = 8):
#         super().__init__()
#         hidden = max(reduction, channels // 4)
#         self.fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, hidden, 1, bias=False), nn.GELU(),
#             nn.Conv2d(hidden, channels, 1, bias=False), nn.Sigmoid()
#         )
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         return z * self.fc(z)
#
#
# class SKMerge(nn.Module):
#     def __init__(self, channels: int, n_branches: int = 3, reduction: int = 8, tau: float = 1.5):
#         super().__init__()
#         hidden = max(reduction, channels // 4)
#         self.squeeze = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, hidden, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(hidden, n_branches * channels, 1, bias=False),
#         )
#         self.n = n_branches
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#         self.tau = tau
#
#     def forward(self, feats: List[torch.Tensor], w_prior: torch.Tensor = None):
#         B, C, _, _ = feats[0].shape
#         s = sum(feats) / self.n
#         a = self.squeeze(s).view(B, self.n, C, 1, 1)
#         if w_prior is not None:
#             if w_prior.dim() == 4:      # [B,1,1,1]
#                 w_prior = w_prior.view(B, 1, 1, 1, 1).expand(B, self.n, C, 1, 1)
#             elif w_prior.dim() == 5 and w_prior.size(1) == 1:  # [B,1,1,1,1]
#                 w_prior = w_prior.expand(B, self.n, C, 1, 1)
#             a = a + self.alpha * w_prior
#         w = torch.softmax(a / self.tau, dim=1)
#         out = sum(w[:, i] * feats[i] for i in range(self.n))
#         return out
#
#
# # ===== 带开关的 Inception2D 残差块 =====
# class Inception2dResBlockAdv(nn.Module):
#     def __init__(self, channels: int, use_sk: bool, use_se: bool, se_strength: float,
#                  use_res_scale: bool, sk_tau: float, drop: float = 0.0):
#         super().__init__()
#         C = channels
#         g = _best_gn_groups(C)
#         self.pre = nn.GroupNorm(g, C)
#
#         self.b3 = nn.Sequential(nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
#                                 nn.GroupNorm(g, C), nn.GELU())
#         self.b5 = nn.Sequential(nn.Conv2d(C, C, 5, padding=2, groups=C, bias=False),
#                                 nn.GroupNorm(g, C), nn.GELU())
#         self.bDil = nn.Sequential(nn.Conv2d(C, C, 3, padding=2, dilation=2, groups=C, bias=False),
#                                   nn.GroupNorm(g, C), nn.GELU())
#
#         self.use_sk = bool(use_sk)
#         self.sk = SKMerge(C, n_branches=3, reduction=max(8, C // 4), tau=sk_tau) if self.use_sk else None
#
#         self.merge_pw = nn.Conv2d(C, C, 1, bias=False)
#         self.merge_gn = nn.GroupNorm(g, C)
#         self.act = nn.GELU()
#         self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
#
#         self.dw = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
#         self.pw = nn.Conv2d(C, C, 1, bias=False)
#         self.gn2 = nn.GroupNorm(g, C)
#
#         self.use_se = bool(use_se)
#         self.se_strength = float(se_strength)
#         self.se = SEBlock(C, reduction=max(8, C // 4)) if self.use_se else None
#
#         self.use_res_scale = bool(use_res_scale)
#         self.res_scale = ResidualScale(C, init=1e-3) if self.use_res_scale else nn.Identity()
#
#     def forward(self, z: torch.Tensor, w_prior_branch: torch.Tensor = None) -> torch.Tensor:
#         x_in = self.pre(z)
#         f1, f2, f3 = self.b3(x_in), self.b5(x_in), self.bDil(x_in)
#         if self.use_sk:
#             x = self.sk([f1, f2, f3], w_prior=w_prior_branch)
#         else:
#             x = (f1 + f2 + f3) / 3.0
#
#         x = self.merge_pw(x)
#         x = self.merge_gn(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.dw(x)
#         x = self.pw(x)
#         x = self.gn2(x)
#         x = self.act(x)
#
#         if self.use_se and self.se_strength > 0:
#             x = x * (self.se(x).pow(self.se_strength))
#
#         x = self.res_scale(x) if self.use_res_scale else x
#         return z + x
#
#
# # ===== 带开关的 cyc 轴大核 conv1d =====
# class CycLargeKernelConv1dAdv(nn.Module):
#     def __init__(self, channels: int, k: int, use_res_scale: bool):
#         super().__init__()
#         C = channels
#         g = _best_gn_groups(C)
#         self.k_cfg = int(k)
#         self.dw3 = nn.Conv1d(C, C, 3, padding=1, groups=C, bias=False)
#         self.pw = nn.Conv1d(C, C, 1, bias=False)
#         self.norm = nn.GroupNorm(g, C)
#         self.act = nn.GELU()
#         self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(C, C, 1, bias=False), nn.Sigmoid())
#         self.use_res_scale = bool(use_res_scale)
#         self.res_scale = ResidualScale(C, init=1e-3) if self.use_res_scale else nn.Identity()
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         # z: [B,C,cyc,p]
#         u = z.mean(dim=-1)  # [B,C,cyc]
#         cyc = u.size(-1)
#         if cyc <= 1:
#             return z
#         k_eff = max(3, min(self.k_cfg, 2 * cyc - 1))
#         rep = max(1, (k_eff // 3))
#         x = u
#         for _ in range(rep):
#             x = self.dw3(x)
#         x = self.pw(x)
#         x = self.norm(x)
#         x = self.act(x)
#         g = self.gate(u)
#         x = x * g
#         x = x.unsqueeze(-1).expand_as(z)  # [B,C,cyc,p]
#         x = self.res_scale(x) if self.use_res_scale else x
#         return z + x
#
#
# # ===== 新增：局部-全局分支（非周期路径）=====
# class LocalGlobalBranch(nn.Module):
#     """
#     不依赖周期假设的特征提取分支
#     - 局部：深度可分离卷积捕获短程模式
#     - 全局：简化自注意力或全局平均
#     """
#
#     def __init__(self, channels: int, local_kernel: int = 7, use_global_attn: bool = False):
#         super().__init__()
#         C = channels
#         g = _best_gn_groups(C)
#
#         # 局部路径：多尺度深度可分离卷积
#         self.local_conv3 = nn.Sequential(
#             nn.Conv1d(C, C, 3, padding=1, groups=C, bias=False),
#             nn.GroupNorm(g, C), nn.GELU()
#         )
#         self.local_conv_large = nn.Sequential(
#             nn.Conv1d(C, C, local_kernel, padding=local_kernel // 2, groups=C, bias=False),
#             nn.GroupNorm(g, C), nn.GELU()
#         )
#
#         # 融合局部特征
#         self.local_merge = nn.Sequential(
#             nn.Conv1d(C * 2, C, 1, bias=False),
#             nn.GroupNorm(g, C), nn.GELU()
#         )
#
#         # 全局路径
#         self.use_global_attn = use_global_attn
#         if use_global_attn:
#             self.global_attn = SimplifiedAttention(C, num_heads=1)
#         else:
#             hidden = max(8, C // 2)
#             self.global_pool = nn.Sequential(
#                 nn.AdaptiveAvgPool1d(1),
#                 nn.Conv1d(C, hidden, 1, bias=False), nn.GELU(),
#                 nn.Conv1d(hidden, C, 1, bias=False), nn.Sigmoid()
#             )
#
#         # 残差缩放（针对 3D 特征）
#         self.res_scale = ResidualScale1D(C, init=1e-3)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B,T,C] -> [B,T,C]
#         """
#         x_nct = x.permute(0, 2, 1)  # [B,C,T]
#
#         # 局部特征提取
#         local3 = self.local_conv3(x_nct)
#         local_large = self.local_conv_large(x_nct)
#         local_feat = self.local_merge(torch.cat([local3, local_large], dim=1))  # [B,C,T]
#
#         # 全局特征提取
#         if self.use_global_attn:
#             local_feat_ntc = local_feat.permute(0, 2, 1)  # [B,T,C]
#             global_feat = self.global_attn(local_feat_ntc)  # [B,T,C]
#             global_feat = global_feat.permute(0, 2, 1)  # [B,C,T]
#         else:
#             gate = self.global_pool(local_feat)  # [B,C,1]
#             global_feat = local_feat * gate  # [B,C,T]
#
#         # 残差连接（保持 3D）
#         out = x_nct + self.res_scale(global_feat)  # [B,C,T]
#         return out.permute(0, 2, 1)  # [B,T,C]
#
#
# class SimplifiedAttention(nn.Module):
#     """简化的单头自注意力（降低计算量）"""
#
#     def __init__(self, channels: int, num_heads: int = 1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = channels // num_heads
#         assert self.head_dim * num_heads == channels
#
#         self.qkv = nn.Linear(channels, channels * 3, bias=False)
#         self.proj = nn.Linear(channels, channels, bias=False)
#         self.scale = self.head_dim ** -0.5
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """x: [B,T,C]"""
#         B, T, C = x.shape
#         qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, T, head_dim]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
#         attn = F.softmax(attn, dim=-1)
#
#         out = (attn @ v).transpose(1, 2).reshape(B, T, C)  # [B,T,C]
#         out = self.proj(out)
#         return out
#
#
# # ====== 工具：频域置信度与正交残差（修复版） ======
# def _period_confidence_from_wprior(w_prior: torch.Tensor, eps: float = 1e-8):
#     """
#     w_prior: [B, k]（自相关 top-k 的 softmax 权重）
#     返回 conf∈[0,1]，越接近1表示周期越明确
#     """
#     H = -(w_prior * (w_prior + eps).log()).sum(dim=1)    # [B]
#     k = w_prior.size(1)
#     if k <= 1:
#         return torch.zeros_like(H)
#     H_max = torch.log(torch.tensor(k, dtype=w_prior.dtype, device=w_prior.device) + eps)
#     conf = 1.0 - (H / (H_max + eps))
#     return conf.clamp(0.0, 1.0)  # [B]
#
#
# def _orth_residual(local_res: torch.Tensor, periodic_res: torch.Tensor, eps=1e-6):
#     """
#     对时间维做投影消除：local - proj(local on periodic)
#     形状均为 [B,T,C]
#     """
#     num = (local_res * periodic_res).sum(dim=1, keepdim=True)            # [B,1,C]
#     den = (periodic_res * periodic_res).sum(dim=1, keepdim=True).add(eps)
#     proj = (num / den) * periodic_res                                   # [B,T,C]
#     return local_res - proj
#
#
# # ===== 带双路径的 TimesBlock（主干+门控旁路） =====
# class AdvTimesBlock(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         # 基本超参
#         self.k = configs.top_k
#         self.kernel = configs.moving_avg
#         self.d_model = configs.d_model
#         self.min_p = getattr(configs, 'min_period', 3)
#         self.drop2d = getattr(configs, 'conv2d_dropout', 0.0)
#
#         # 开关
#         self.use_series_decomp = bool(getattr(configs, 'use_series_decomp', 1))
#         self.use_sk = bool(getattr(configs, 'use_sk', 1))
#         self.use_se = bool(getattr(configs, 'use_se', 0))
#         self.se_strength = float(getattr(configs, 'se_strength', 0.0))
#         self.use_cyc_conv1d = bool(getattr(configs, 'use_cyc_conv1d', 1))
#         self.cyc_k = int(getattr(configs, 'cyc_conv_kernel', 9))
#         self.use_gate_mlp = bool(getattr(configs, 'use_gate_mlp', 1))
#         self.use_res_scale = bool(getattr(configs, 'use_res_scale', 1))
#         self.reflect_pad = bool(getattr(configs, 'reflect_pad', 1))
#         self.sk_tau = float(getattr(configs, 'sk_tau', 1.5))
#
#         # 双路径
#         self.use_dual_path = bool(getattr(configs, 'use_dual_path', 0))
#         self.local_kernel = int(getattr(configs, 'local_kernel', 7))
#         self.use_global_attn = bool(getattr(configs, 'use_global_attn', 0))
#
#         # 组件
#         self.decomp = SeriesDecomp(self.kernel) if self.use_series_decomp else None
#
#         # 周期主干
#         self.block2d = Inception2dResBlockAdv(
#             self.d_model, use_sk=self.use_sk, use_se=self.use_se,
#             se_strength=self.se_strength, use_res_scale=self.use_res_scale,
#             sk_tau=self.sk_tau, drop=self.drop2d
#         )
#         self.cyc1d = (CycLargeKernelConv1dAdv(self.d_model, k=self.cyc_k, use_res_scale=self.use_res_scale)
#                       if self.use_cyc_conv1d else None)
#
#         # 局部-全局旁路
#         self.local_global_branch = (LocalGlobalBranch(self.d_model,
#                                                       local_kernel=self.local_kernel,
#                                                       use_global_attn=self.use_global_attn)
#                                     if self.use_dual_path else None)
#
#         # ===== 动态门控（per-channel），默认偏闭；支持频域先验 + warmup =====
#         if self.use_dual_path:
#             hidden = max(16, self.d_model // 4)
#             self.gating_network = nn.Sequential(
#                 nn.Linear(self.d_model, hidden),
#                 nn.GELU(),
#                 nn.Linear(hidden, self.d_model)
#             )
#             # 初始化：让门默认偏闭（对 Hell 友好）
#             nn.init.zeros_(self.gating_network[0].weight)
#             nn.init.zeros_(self.gating_network[0].bias)
#             nn.init.xavier_uniform_(self.gating_network[2].weight)
#             nn.init.constant_(self.gating_network[2].bias, -2.0)
#
#             # 频域先验强度/阈值 + 旁路幅度系数 + 可选 warmup（按 epoch 粗估）
#             self.gate_alpha = float(getattr(configs, 'gate_alpha', 4.0))
#             self.gate_tau   = float(getattr(configs, 'gate_tau',   0.5))
#             self.dual_beta  = float(getattr(configs, 'dual_beta',  0.3))
#             self.gate_warmup = int(getattr(configs, 'gate_warmup', 0))  # 0 表示不启用 warmup
#             self._fw_steps = 0  # forward 步数计数器（warmup 用）
#
#         hidden = max(8, self.d_model // 4)
#         self.gate_mlp = (nn.Sequential(nn.Linear(self.d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))
#                          if self.use_gate_mlp else None)
#         self.eps = 1e-8
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B,T,C=d_model]
#         """
#         B, T, C = x.shape
#         assert C == self.d_model, "Input last dim must equal d_model"
#
#         # ===== 周期主干 =====
#         if self.decomp is not None:
#             s, t = self.decomp(x)
#         else:
#             s, t = x, 0.0
#
#         idx, w_prior = auto_correlation(s, self.k, min_p=self.min_p, max_p=T // 2)  # [B,k],[B,k]
#
#         agg_num = torch.zeros_like(x)
#         agg_den = x.new_zeros(B)
#
#         unique_p = torch.unique(idx)
#         for pv in unique_p.tolist():
#             mask = (idx == pv)
#             if not mask.any():
#                 continue
#             b_idx, j_idx = mask.nonzero(as_tuple=True)
#             m = b_idx.numel()
#
#             sb = s[b_idx]  # [m,T,C]
#             wb = w_prior[b_idx, j_idx].view(m, 1, 1)  # [m,1,1]
#
#             z, _T = _fold_2d(sb, int(pv), reflect_pad=self.reflect_pad)  # [m,C,cyc,p]
#             z = self.block2d(z)  # [m,C,cyc,p]
#             if self.cyc1d is not None:
#                 z = self.cyc1d(z)
#             y = _unfold_1d(z, _T)  # [m,T,C]
#
#             if self.gate_mlp is not None:
#                 gate = torch.sigmoid(self.gate_mlp(y.mean(dim=1)))  # [m,1]
#                 score = torch.log(wb + self.eps) + torch.log(gate.view(m, 1, 1) + self.eps)
#             else:
#                 score = torch.log(wb + self.eps)
#             score_exp = torch.exp(score)  # [m,1,1]
#
#             contrib = y * score_exp  # [m,T,C]
#             agg_num.index_add_(0, b_idx, contrib)
#             agg_den.index_add_(0, b_idx, score_exp.view(-1))
#
#         agg = agg_num / (agg_den.view(B, 1, 1) + self.eps)
#
#         # 主干输出：x + periodic_residual
#         periodic_out = agg + (t if isinstance(t, torch.Tensor) else 0.0)
#         periodic_out = periodic_out + (x - (t if isinstance(t, torch.Tensor) else 0.0))
#         periodic_residual = periodic_out - x  # [B,T,C]
#
#         # ===== 旁路：Local-Global（可选）=====
#         if self.use_dual_path and self.local_global_branch is not None:
#             local_global_out = self.local_global_branch(x)      # [B,T,C]
#             local_residual = local_global_out - x               # [B,T,C]
#
#             # 正交化：只保留互补信息（不反向影响主干）
#             local_residual = _orth_residual(local_residual, periodic_residual.detach())
#
#             # 频域先验（无梯度，周期越强门越关）
#             with torch.no_grad():
#                 conf = _period_confidence_from_wprior(w_prior)  # [B]
#
#             # learned gate（per-channel）+ 正确的“减号”先验偏置
#             x_global = x.mean(dim=1)                            # [B,C]
#             g_logit = self.gating_network(x_global)             # [B,C]
#             g_logit = g_logit - self.gate_alpha * (conf - self.gate_tau).unsqueeze(-1)
#             gate = torch.sigmoid(g_logit).unsqueeze(1)          # [B,1,C]
#
#             # 可选：warmup 冻结旁路前若干 epoch（粗估 steps_per_epoch=374，来自你的 Hell 日志）
#             if self.training and self.gate_warmup > 0:
#                 self._fw_steps += 1
#                 steps_per_epoch = 374
#                 if self._fw_steps < steps_per_epoch * self.gate_warmup:
#                     gate = gate * 0.0
#
#             # 限幅叠加：防过补
#             out = periodic_out + gate * (self.dual_beta * local_residual)
#             return out
#
#         # 仅主干
#         return periodic_out
#
#
# # ===== 继承 MyModule.Model，并替换 blocks =====
# class Model(MyModule.Model):
#     """
#     继承你的 MyModule.Model：
#     - super().__init__(configs) 后，直接替换 self.blocks 为 AdvTimesBlock 的 ModuleList
#     - 其它（project / norms / attention_pool / classifier）完全沿用父类实现
#     - 因为 forward 会遍历 self.blocks，所以无需重写 forward
#     """
#
#     def __init__(self, configs):
#         super().__init__(configs)
#         try:
#             self.blocks = nn.ModuleList([AdvTimesBlock(configs) for _ in range(configs.e_layers)])
#             print(f"[TimesNetAdv] Successfully initialized with dual_path={getattr(configs, 'use_dual_path', 0)}")
#         except Exception as e:
#             print(f"[TimesNetAdv] WARNING: fallback to MyModule blocks due to: {e}")
#             pass



# ===================版本2===================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, List
#
# # ===== 继承你的 MyModule =====
# from models import MyModule
#
#
# # ===== 基础工具 =====
# def _best_gn_groups(C: int) -> int:
#     for g in [32, 16, 8, 4, 2, 1]:
#         if C % g == 0:
#             return g
#     return 1
#
#
# class ResidualScale(nn.Module):
#     """用于 4D 张量（如 [B,C,H,W] / [B,C,cyc,p]）的残差缩放"""
#     def __init__(self, channels: int, init: float = 1e-3):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * init)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * self.gamma
#
#
# class ResidualScale1D(nn.Module):
#     """用于 3D 张量 [B,C,T] 的残差缩放，避免把 3D 广播成 4D"""
#     def __init__(self, channels: int, init: float = 1e-3):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(1, channels, 1) * init)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * self.gamma
#
#
# class SeriesDecomp(nn.Module):
#     def __init__(self, kernel_size: int):
#         super().__init__()
#         self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
#                                      padding=(kernel_size - 1) // 2, count_include_pad=False)
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x_nct = x.permute(0, 2, 1)
#         trend = self.avg_pool(x_nct).permute(0, 2, 1)
#         seasonal = x - trend
#         return seasonal, trend
#
#
# def auto_correlation(x: torch.Tensor, k: int, min_p: int = 3, max_p: int = None):
#     B, T, C = x.shape
#     dtype = x.dtype
#     if max_p is None:
#         max_p = max(1, T // 2)
#     max_p = max(min_p, min(max_p, T - 1))
#     Xf = torch.fft.rfft(x, dim=1)
#     P = (Xf * torch.conj(Xf)).real
#     Pm = P.mean(dim=-1)
#     r = torch.fft.irfft(Pm, n=T, dim=1)
#
#     mask = torch.ones_like(r, dtype=torch.bool)
#     mask[:, 0] = False
#     if min_p > 1: mask[:, 1:min_p] = False
#     if max_p + 1 < T: mask[:, max_p + 1:] = False
#
#     very_neg = torch.finfo(dtype).min / 8 if dtype.is_floating_point else -1e9
#     r_masked = torch.where(mask, r, torch.full_like(r, very_neg))
#
#     p0 = max(min_p, min(max_p, max(2, T // 4)))
#     all_bad = (mask.sum(dim=1) == 0)
#     if all_bad.any():
#         r_masked[all_bad] = very_neg
#         r_masked[all_bad, p0] = 0.0
#
#     k_eff = min(k, max_p - min_p + 1)
#     vals, idx = torch.topk(r_masked, k=k_eff, dim=1)
#     if k_eff < k:
#         pad_n = k - k_eff
#         idx = torch.cat([idx, idx[:, -1:].expand(B, pad_n)], dim=1)
#         vals = torch.cat([vals, vals[:, -1:].expand(B, pad_n)], dim=1)
#     w = F.softmax(vals, dim=1)
#     return idx, w
#
#
# def _fold_2d(x_1d: torch.Tensor, p: int, reflect_pad: bool) -> Tuple[torch.Tensor, int]:
#     """x_1d:[b,T,C] -> z:[b,C,cyc,p]"""
#     b, T, C = x_1d.shape
#     pad = ((T + p - 1) // p) * p - T
#     if pad > 0:
#         x_nct = x_1d.permute(0, 2, 1).contiguous()
#         mode = 'reflect' if reflect_pad else 'constant'
#         x_nct = F.pad(x_nct, (0, pad), mode=mode, value=0.0 if not reflect_pad else 0.0)
#         x_1d = x_nct.permute(0, 2, 1).contiguous()
#     T_new = x_1d.shape[1]
#     cyc = T_new // p
#     z = x_1d.reshape(b, cyc, p, C).permute(0, 3, 1, 2).contiguous()
#     return z, T
#
#
# def _unfold_1d(z_2d: torch.Tensor, T: int) -> torch.Tensor:
#     """z_2d:[b,C,cyc,p] -> y:[b,T,C]"""
#     b, C, cyc, p = z_2d.shape
#     y = z_2d.permute(0, 2, 3, 1).reshape(b, cyc * p, C)[:, :T, :].contiguous()
#     return y
#
#
# class SEBlock(nn.Module):
#     def __init__(self, channels: int, reduction: int = 8):
#         super().__init__()
#         hidden = max(reduction, channels // 4)
#         self.fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, hidden, 1, bias=False), nn.GELU(),
#             nn.Conv2d(hidden, channels, 1, bias=False), nn.Sigmoid()
#         )
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         return z * self.fc(z)
#
#
# class SKMerge(nn.Module):
#     def __init__(self, channels: int, n_branches: int = 3, reduction: int = 8, tau: float = 1.5):
#         super().__init__()
#         hidden = max(reduction, channels // 4)
#         self.squeeze = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, hidden, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(hidden, n_branches * channels, 1, bias=False),
#         )
#         self.n = n_branches
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#         self.tau = tau
#
#     def forward(self, feats: List[torch.Tensor], w_prior: torch.Tensor = None):
#         B, C, _, _ = feats[0].shape
#         s = sum(feats) / self.n
#         a = self.squeeze(s).view(B, self.n, C, 1, 1)
#         if w_prior is not None:
#             if w_prior.dim() == 4:
#                 w_prior = w_prior.view(B, 1, 1, 1, 1).expand(B, self.n, C, 1, 1)
#             elif w_prior.dim() == 5 and w_prior.size(1) == 1:
#                 w_prior = w_prior.expand(B, self.n, C, 1, 1)
#             a = a + self.alpha * w_prior
#         w = torch.softmax(a / self.tau, dim=1)
#         out = sum(w[:, i] * feats[i] for i in range(self.n))
#         return out
#
#
# # ===== 带开关的 Inception2D 残差块 =====
# class Inception2dResBlockAdv(nn.Module):
#     def __init__(self, channels: int, use_sk: bool, use_se: bool, se_strength: float,
#                  use_res_scale: bool, sk_tau: float, drop: float = 0.0):
#         super().__init__()
#         C = channels
#         g = _best_gn_groups(C)
#         self.pre = nn.GroupNorm(g, C)
#
#         self.b3 = nn.Sequential(nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
#                                 nn.GroupNorm(g, C), nn.GELU())
#         self.b5 = nn.Sequential(nn.Conv2d(C, C, 5, padding=2, groups=C, bias=False),
#                                 nn.GroupNorm(g, C), nn.GELU())
#         self.bDil = nn.Sequential(nn.Conv2d(C, C, 3, padding=2, dilation=2, groups=C, bias=False),
#                                   nn.GroupNorm(g, C), nn.GELU())
#
#         self.use_sk = bool(use_sk)
#         self.sk = SKMerge(C, n_branches=3, reduction=max(8, C // 4), tau=sk_tau) if self.use_sk else None
#
#         self.merge_pw = nn.Conv2d(C, C, 1, bias=False)
#         self.merge_gn = nn.GroupNorm(g, C)
#         self.act = nn.GELU()
#         self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
#
#         self.dw = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
#         self.pw = nn.Conv2d(C, C, 1, bias=False)
#         self.gn2 = nn.GroupNorm(g, C)
#
#         self.use_se = bool(use_se)
#         self.se_strength = float(se_strength)
#         self.se = SEBlock(C, reduction=max(8, C // 4)) if self.use_se else None
#
#         self.use_res_scale = bool(use_res_scale)
#         self.res_scale = ResidualScale(C, init=1e-3) if self.use_res_scale else nn.Identity()
#
#     def forward(self, z: torch.Tensor, w_prior_branch: torch.Tensor = None) -> torch.Tensor:
#         x_in = self.pre(z)
#         f1, f2, f3 = self.b3(x_in), self.b5(x_in), self.bDil(x_in)
#         if self.use_sk:
#             x = self.sk([f1, f2, f3], w_prior=w_prior_branch)
#         else:
#             x = (f1 + f2 + f3) / 3.0
#
#         x = self.merge_pw(x)
#         x = self.merge_gn(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.dw(x)
#         x = self.pw(x)
#         x = self.gn2(x)
#         x = self.act(x)
#
#         if self.use_se and self.se_strength > 0:
#             x = x * (self.se(x).pow(self.se_strength))
#
#         x = self.res_scale(x) if self.use_res_scale else x
#         return z + x
#
#
# # ===== 带开关的 cyc 轴大核 conv1d =====
# class CycLargeKernelConv1dAdv(nn.Module):
#     def __init__(self, channels: int, k: int, use_res_scale: bool):
#         super().__init__()
#         C = channels
#         g = _best_gn_groups(C)
#         self.k_cfg = int(k)
#         self.dw3 = nn.Conv1d(C, C, 3, padding=1, groups=C, bias=False)
#         self.pw = nn.Conv1d(C, C, 1, bias=False)
#         self.norm = nn.GroupNorm(g, C)
#         self.act = nn.GELU()
#         self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(C, C, 1, bias=False), nn.Sigmoid())
#         self.use_res_scale = bool(use_res_scale)
#         self.res_scale = ResidualScale(C, init=1e-3) if self.use_res_scale else nn.Identity()
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         # z: [B,C,cyc,p]
#         u = z.mean(dim=-1)  # [B,C,cyc]
#         cyc = u.size(-1)
#         if cyc <= 1:
#             return z
#         k_eff = max(3, min(self.k_cfg, 2 * cyc - 1))
#         rep = max(1, (k_eff // 3))
#         x = u
#         for _ in range(rep):
#             x = self.dw3(x)
#         x = self.pw(x)
#         x = self.norm(x)
#         x = self.act(x)
#         g = self.gate(u)
#         x = x * g
#         x = x.unsqueeze(-1).expand_as(z)  # [B,C,cyc,p]
#         x = self.res_scale(x) if self.use_res_scale else x
#         return z + x
#
#
# # ===== 新增：局部-全局分支（非周期路径）=====
# class LocalGlobalBranch(nn.Module):
#     """
#     不依赖周期假设的特征提取分支
#     - 局部：深度可分离卷积捕获短程模式
#     - 全局：简化自注意力或全局平均
#     """
#
#     def __init__(self, channels: int, local_kernel: int = 7, use_global_attn: bool = False):
#         super().__init__()
#         C = channels
#         g = _best_gn_groups(C)
#
#         # 局部路径：多尺度深度可分离卷积
#         self.local_conv3 = nn.Sequential(
#             nn.Conv1d(C, C, 3, padding=1, groups=C, bias=False),
#             nn.GroupNorm(g, C), nn.GELU()
#         )
#         self.local_conv_large = nn.Sequential(
#             nn.Conv1d(C, C, local_kernel, padding=local_kernel // 2, groups=C, bias=False),
#             nn.GroupNorm(g, C), nn.GELU()
#         )
#
#         # 融合局部特征
#         self.local_merge = nn.Sequential(
#             nn.Conv1d(C * 2, C, 1, bias=False),
#             nn.GroupNorm(g, C), nn.GELU()
#         )
#
#         # 全局路径
#         self.use_global_attn = use_global_attn
#         if use_global_attn:
#             # 简化的全局自注意力（单头，降低复杂度）
#             self.global_attn = SimplifiedAttention(C, num_heads=1)
#         else:
#             # 全局平均池化 + MLP（更轻量）
#             hidden = max(8, C // 2)
#             self.global_pool = nn.Sequential(
#                 nn.AdaptiveAvgPool1d(1),
#                 nn.Conv1d(C, hidden, 1, bias=False), nn.GELU(),
#                 nn.Conv1d(hidden, C, 1, bias=False), nn.Sigmoid()
#             )
#
#         # 残差缩放（针对 3D 特征）
#         self.res_scale = ResidualScale1D(C, init=1e-3)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B,T,C] -> [B,T,C]
#         """
#         x_nct = x.permute(0, 2, 1)  # [B,C,T]
#
#         # 局部特征提取
#         local3 = self.local_conv3(x_nct)
#         local_large = self.local_conv_large(x_nct)
#         local_feat = self.local_merge(torch.cat([local3, local_large], dim=1))  # [B,C,T]
#
#         # 全局特征提取
#         if self.use_global_attn:
#             local_feat_ntc = local_feat.permute(0, 2, 1)  # [B,T,C]
#             global_feat = self.global_attn(local_feat_ntc)  # [B,T,C]
#             global_feat = global_feat.permute(0, 2, 1)  # [B,C,T]
#         else:
#             # 全局门控
#             gate = self.global_pool(local_feat)  # [B,C,1]
#             global_feat = local_feat * gate  # [B,C,T]
#
#         # 残差连接（保持 3D，不再 squeeze）
#         out = x_nct + self.res_scale(global_feat)  # [B,C,T]
#         return out.permute(0, 2, 1)  # [B,T,C]
#
#
# class SimplifiedAttention(nn.Module):
#     """简化的单头自注意力（降低计算量）"""
#
#     def __init__(self, channels: int, num_heads: int = 1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = channels // num_heads
#         assert self.head_dim * num_heads == channels
#
#         self.qkv = nn.Linear(channels, channels * 3, bias=False)
#         self.proj = nn.Linear(channels, channels, bias=False)
#         self.scale = self.head_dim ** -0.5
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """x: [B,T,C]"""
#         B, T, C = x.shape
#         qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, T, head_dim]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
#         attn = F.softmax(attn, dim=-1)
#
#         out = (attn @ v).transpose(1, 2).reshape(B, T, C)  # [B,T,C]
#         out = self.proj(out)
#         return out
#
#
# # ===== 带双路径的 TimesBlock =====
# class AdvTimesBlock(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         # 基本超参
#         self.k = configs.top_k
#         self.kernel = configs.moving_avg
#         self.d_model = configs.d_model
#         self.min_p = getattr(configs, 'min_period', 3)
#         self.drop2d = getattr(configs, 'conv2d_dropout', 0.0)
#
#         # 开关
#         self.use_series_decomp = bool(getattr(configs, 'use_series_decomp', 1))
#         self.use_sk = bool(getattr(configs, 'use_sk', 1))
#         self.use_se = bool(getattr(configs, 'use_se', 0))
#         self.se_strength = float(getattr(configs, 'se_strength', 0.0))
#         self.use_cyc_conv1d = bool(getattr(configs, 'use_cyc_conv1d', 1))
#         self.cyc_k = int(getattr(configs, 'cyc_conv_kernel', 9))
#         self.use_gate_mlp = bool(getattr(configs, 'use_gate_mlp', 1))
#         self.use_res_scale = bool(getattr(configs, 'use_res_scale', 1))
#         self.reflect_pad = bool(getattr(configs, 'reflect_pad', 1))
#         self.sk_tau = float(getattr(configs, 'sk_tau', 1.5))
#
#         # 【新增】双路径开关
#         self.use_dual_path = bool(getattr(configs, 'use_dual_path', 0))
#         self.local_kernel = int(getattr(configs, 'local_kernel', 7))
#         self.use_global_attn = bool(getattr(configs, 'use_global_attn', 0))
#         self.dual_fusion_mode = str(
#             getattr(configs, 'dual_fusion_mode', 'learnable'))  # 'learnable', 'fixed', 'adaptive'
#
#         # 组件
#         self.decomp = SeriesDecomp(self.kernel) if self.use_series_decomp else None
#
#         # 周期路径（原有）
#         self.block2d = Inception2dResBlockAdv(
#             self.d_model, use_sk=self.use_sk, use_se=self.use_se,
#             se_strength=self.se_strength, use_res_scale=self.use_res_scale,
#             sk_tau=self.sk_tau, drop=self.drop2d
#         )
#         self.cyc1d = (CycLargeKernelConv1dAdv(self.d_model, k=self.cyc_k, use_res_scale=self.use_res_scale)
#                       if self.use_cyc_conv1d else None)
#
#         # 【新增】局部-全局路径
#         self.local_global_branch = (LocalGlobalBranch(self.d_model,
#                                                       local_kernel=self.local_kernel,
#                                                       use_global_attn=self.use_global_attn)
#                                     if self.use_dual_path else None)
#
#         # 【新增】双路径融合权重
#         if self.use_dual_path:
#             if self.dual_fusion_mode == 'learnable':
#                 # 可学习固定权重
#                 self.fusion_weight = nn.Parameter(torch.tensor([2.0, -2.0]))
#                 # self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
#             elif self.dual_fusion_mode == 'adaptive':
#                 # 内容自适应融合（基于输入特征动态决定）
#                 hidden = max(8, self.d_model // 4)
#                 self.fusion_gate = nn.Sequential(
#                     nn.Linear(self.d_model, hidden),
#                     nn.GELU(),
#                     nn.Linear(hidden, 2),
#                     nn.Softmax(dim=-1)
#                 )
#             # 'fixed' 模式：固定 0.5:0.5，无需额外参数
#
#         hidden = max(8, self.d_model // 4)
#         self.gate_mlp = (nn.Sequential(nn.Linear(self.d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))
#                          if self.use_gate_mlp else None)
#         self.eps = 1e-8
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B,T,C=d_model]
#         """
#         B, T, C = x.shape
#         assert C == self.d_model, "Input last dim must equal d_model"
#
#         # ===== 路径A：周期建模路径 =====
#         # 分解（可关）
#         if self.decomp is not None:
#             s, t = self.decomp(x)
#         else:
#             s, t = x, 0.0
#
#         # 周期候选
#         idx, w_prior = auto_correlation(s, self.k, min_p=self.min_p, max_p=T // 2)  # [B,k],[B,k]
#
#         agg_num = torch.zeros_like(x)
#         agg_den = x.new_zeros(B)
#
#         # 以 unique(p) 聚合，减少重复折叠
#         unique_p = torch.unique(idx)
#         for pv in unique_p.tolist():
#             mask = (idx == pv)
#             if not mask.any():
#                 continue
#             b_idx, j_idx = mask.nonzero(as_tuple=True)
#             m = b_idx.numel()
#
#             sb = s[b_idx]  # [m,T,C]
#             wb = w_prior[b_idx, j_idx].view(m, 1, 1)  # [m,1,1]
#
#             # 折叠到 2D（可选反射填充）
#             z, _T = _fold_2d(sb, int(pv), reflect_pad=self.reflect_pad)  # [m,C,cyc,p]
#
#             # 多尺度 2D +（可选 SK/SE/残差缩放）
#             z = self.block2d(z)  # [m,C,cyc,p]
#
#             # 可选：cyc 轴大核卷积
#             if self.cyc1d is not None:
#                 z = self.cyc1d(z)
#
#             # 展回 1D
#             y = _unfold_1d(z, _T)  # [m,T,C]
#
#             # 融合权重：w_prior ×（可选 gate）
#             if self.gate_mlp is not None:
#                 gate = torch.sigmoid(self.gate_mlp(y.mean(dim=1)))  # [m,1]
#                 score = torch.log(wb + self.eps) + torch.log(gate.view(m, 1, 1) + self.eps)
#             else:
#                 score = torch.log(wb + self.eps)
#             score_exp = torch.exp(score)  # [m,1,1]
#
#             contrib = y * score_exp  # [m,T,C]
#             agg_num.index_add_(0, b_idx, contrib)
#             agg_den.index_add_(0, b_idx, score_exp.view(-1))
#
#         agg = agg_num / (agg_den.view(B, 1, 1) + self.eps)
#
#         # 加回趋势，并保留季节残差（若没分解则 t=0）
#         periodic_out = agg + (t if isinstance(t, torch.Tensor) else 0.0)
#         periodic_out = periodic_out + (x - (t if isinstance(t, torch.Tensor) else 0.0))
#
#         # ===== 路径B：局部-全局路径（可选）=====
#         if self.use_dual_path and self.local_global_branch is not None:
#             local_global_out = self.local_global_branch(x)  # [B,T,C]
#
#             # 融合两路输出
#             if self.dual_fusion_mode == 'learnable':
#                 # 固定可学习权重
#                 w = F.softmax(self.fusion_weight, dim=0)
#                 out = w[0] * periodic_out + w[1] * local_global_out
#             elif self.dual_fusion_mode == 'adaptive':
#                 # 内容自适应权重（基于输入全局特征）
#                 x_global = x.mean(dim=1)  # [B,C]
#                 w = self.fusion_gate(x_global)  # [B,2]
#                 out = w[:, 0:1, None] * periodic_out + w[:, 1:2, None] * local_global_out
#             else:  # 'fixed'
#                 out = 0.5 * periodic_out + 0.5 * local_global_out
#
#             return out
#         else:
#             # 仅使用周期路径
#             return periodic_out
#
#
# # ===== 继承 MyModule.Model，并替换 blocks =====
# class Model(MyModule.Model):
#     """
#     继承你的 MyModule.Model：
#     - super().__init__(configs) 后，直接替换 self.blocks 为 AdvTimesBlock 的 ModuleList
#     - 其它（project / norms / attention_pool / classifier）完全沿用父类实现
#     - 因为 forward 会遍历 self.blocks，所以无需重写 forward
#     """
#
#     def __init__(self, configs):
#         super().__init__(configs)
#         # 用带开关的 AdvTimesBlock 替换父类构建的 blocks
#         try:
#             self.blocks = nn.ModuleList([AdvTimesBlock(configs) for _ in range(configs.e_layers)])
#             print(f"[TimesNetAdv] Successfully initialized with dual_path={getattr(configs, 'use_dual_path', 0)}")
#         except Exception as e:
#             print(f"[TimesNetAdv] WARNING: fallback to MyModule blocks due to: {e}")
#             # 出现异常则保留父类 blocks，不影响正常训练/推理
#             pass


# =================版本1=================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, List
#
# # ===== 继承你的 MyModule =====
# from models import MyModule
#
# # ===== 基础工具 =====
# def _best_gn_groups(C: int) -> int:
#     for g in [32, 16, 8, 4, 2, 1]:
#         if C % g == 0:
#             return g
#     return 1
#
# class ResidualScale(nn.Module):
#     def __init__(self, channels: int, init: float = 1e-3):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * init)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * self.gamma
#
# class SeriesDecomp(nn.Module):
#     def __init__(self, kernel_size: int):
#         super().__init__()
#         self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
#                                      padding=(kernel_size - 1) // 2, count_include_pad=False)
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x_nct = x.permute(0, 2, 1)
#         trend = self.avg_pool(x_nct).permute(0, 2, 1)
#         seasonal = x - trend
#         return seasonal, trend
#
# def auto_correlation(x: torch.Tensor, k: int, min_p: int = 3, max_p: int = None):
#     B, T, C = x.shape
#     dtype = x.dtype
#     if max_p is None:
#         max_p = max(1, T // 2)
#     max_p = max(min_p, min(max_p, T - 1))
#     Xf = torch.fft.rfft(x, dim=1)
#     P  = (Xf * torch.conj(Xf)).real
#     Pm = P.mean(dim=-1)
#     r  = torch.fft.irfft(Pm, n=T, dim=1)
#
#     mask = torch.ones_like(r, dtype=torch.bool)
#     mask[:, 0] = False
#     if min_p > 1: mask[:, 1:min_p] = False
#     if max_p + 1 < T: mask[:, max_p + 1:] = False
#
#     very_neg = torch.finfo(dtype).min / 8 if dtype.is_floating_point else -1e9
#     r_masked = torch.where(mask, r, torch.full_like(r, very_neg))
#
#     # p0 = max(min_p, min(max_p, max(2, T // 4)))
#     # if (mask.sum(dim=1) == 0).any():
#     #     r_masked[:] = very_neg
#     #     r_masked[:, p0] = 0.0
#     # 逐样本回退
#     p0 = max(min_p, min(max_p, max(2, T // 4)))
#     all_bad = (mask.sum(dim=1) == 0)
#     if all_bad.any():
#         r_masked[all_bad] = very_neg
#         r_masked[all_bad, p0] = 0.0
#
#     k_eff = min(k, max_p - min_p + 1)
#     vals, idx = torch.topk(r_masked, k=k_eff, dim=1)
#     if k_eff < k:
#         pad_n = k - k_eff
#         idx  = torch.cat([idx,  idx[:, -1:].expand(B, pad_n)], dim=1)
#         vals = torch.cat([vals, vals[:, -1:].expand(B, pad_n)], dim=1)
#     w = F.softmax(vals, dim=1)
#     return idx, w
#
# def _fold_2d(x_1d: torch.Tensor, p: int, reflect_pad: bool) -> Tuple[torch.Tensor, int]:
#     """x_1d:[b,T,C] -> z:[b,C,cyc,p]"""
#     b, T, C = x_1d.shape
#     pad = ((T + p - 1) // p) * p - T
#     if pad > 0:
#         x_nct = x_1d.permute(0, 2, 1).contiguous()
#         mode = 'reflect' if reflect_pad else 'constant'
#         x_nct = F.pad(x_nct, (0, pad), mode=mode, value=0.0 if not reflect_pad else 0.0)
#         x_1d  = x_nct.permute(0, 2, 1).contiguous()
#     T_new = x_1d.shape[1]
#     cyc = T_new // p
#     z = x_1d.reshape(b, cyc, p, C).permute(0, 3, 1, 2).contiguous()
#     return z, T
#
# def _unfold_1d(z_2d: torch.Tensor, T: int) -> torch.Tensor:
#     """z_2d:[b,C,cyc,p] -> y:[b,T,C]"""
#     b, C, cyc, p = z_2d.shape
#     y = z_2d.permute(0, 2, 3, 1).reshape(b, cyc * p, C)[:, :T, :].contiguous()
#     return y
#
# class SEBlock(nn.Module):
#     def __init__(self, channels: int, reduction: int = 8):
#         super().__init__()
#         hidden = max(reduction, channels // 4)
#         self.fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, hidden, 1, bias=False), nn.GELU(),
#             nn.Conv2d(hidden, channels, 1, bias=False), nn.Sigmoid()
#         )
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         return z * self.fc(z)
#
# class SKMerge(nn.Module):
#     def __init__(self, channels: int, n_branches: int = 3, reduction: int = 8, tau: float = 1.5):
#         super().__init__()
#         hidden = max(reduction, channels // 4)
#         self.squeeze = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, hidden, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(hidden, n_branches * channels, 1, bias=False),
#         )
#         self.n = n_branches
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#         self.tau = tau
#     def forward(self, feats: List[torch.Tensor], w_prior: torch.Tensor = None):
#         B, C, _, _ = feats[0].shape
#         s = sum(feats) / self.n
#         a = self.squeeze(s).view(B, self.n, C, 1, 1)
#         if w_prior is not None:
#             if w_prior.dim() == 4:      # [B,1,1,1]
#                 w_prior = w_prior.view(B, 1, 1, 1, 1).expand(B, self.n, C, 1, 1)
#             elif w_prior.dim() == 5 and w_prior.size(1) == 1:  # [B,1,1,1,1]
#                 w_prior = w_prior.expand(B, self.n, C, 1, 1)
#             a = a + self.alpha * w_prior
#         w = torch.softmax(a / self.tau, dim=1)
#         out = sum(w[:, i] * feats[i] for i in range(self.n))
#         return out
#
# # ===== 带开关的 Inception2D 残差块 =====
# class Inception2dResBlockAdv(nn.Module):
#     def __init__(self, channels: int, use_sk: bool, use_se: bool, se_strength: float,
#                  use_res_scale: bool, sk_tau: float, drop: float = 0.0):
#         super().__init__()
#         C = channels; g = _best_gn_groups(C)
#         self.pre = nn.GroupNorm(g, C)
#
#         self.b3   = nn.Sequential(nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
#                                   nn.GroupNorm(g, C), nn.GELU())
#         self.b5   = nn.Sequential(nn.Conv2d(C, C, 5, padding=2, groups=C, bias=False),
#                                   nn.GroupNorm(g, C), nn.GELU())
#         self.bDil = nn.Sequential(nn.Conv2d(C, C, 3, padding=2, dilation=2, groups=C, bias=False),
#                                   nn.GroupNorm(g, C), nn.GELU())
#
#         self.use_sk = bool(use_sk)
#         self.sk     = SKMerge(C, n_branches=3, reduction=max(8, C // 4), tau=sk_tau) if self.use_sk else None
#
#         self.merge_pw = nn.Conv2d(C, C, 1, bias=False)
#         self.merge_gn = nn.GroupNorm(g, C)
#         self.act = nn.GELU()
#         self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
#
#         self.dw  = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
#         self.pw  = nn.Conv2d(C, C, 1, bias=False)
#         self.gn2 = nn.GroupNorm(g, C)
#
#         self.use_se = bool(use_se)
#         self.se_strength = float(se_strength)
#         self.se = SEBlock(C, reduction=max(8, C // 4)) if self.use_se else None
#
#         self.use_res_scale = bool(use_res_scale)
#         self.res_scale = ResidualScale(C, init=1e-3) if self.use_res_scale else nn.Identity()
#
#     def forward(self, z: torch.Tensor, w_prior_branch: torch.Tensor = None) -> torch.Tensor:
#         x_in = self.pre(z)
#         f1, f2, f3 = self.b3(x_in), self.b5(x_in), self.bDil(x_in)
#         if self.use_sk:
#             x = self.sk([f1, f2, f3], w_prior=w_prior_branch)
#         else:
#             x = (f1 + f2 + f3) / 3.0
#
#         x = self.merge_pw(x); x = self.merge_gn(x); x = self.act(x); x = self.drop(x)
#         x = self.dw(x); x = self.pw(x); x = self.gn2(x); x = self.act(x)
#
#         if self.use_se and self.se_strength > 0:
#             x = x * (self.se(x).pow(self.se_strength))
#
#         x = self.res_scale(x) if self.use_res_scale else x
#         return z + x
#
# # ===== 带开关的 cyc 轴大核 conv1d =====
# class CycLargeKernelConv1dAdv(nn.Module):
#     def __init__(self, channels: int, k: int, use_res_scale: bool):
#         super().__init__()
#         C = channels; g = _best_gn_groups(C)
#         self.k_cfg = int(k)
#         self.dw3 = nn.Conv1d(C, C, 3, padding=1, groups=C, bias=False)
#         self.pw  = nn.Conv1d(C, C, 1, bias=False)
#         self.norm = nn.GroupNorm(g, C)
#         self.act  = nn.GELU()
#         self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(C, C, 1, bias=False), nn.Sigmoid())
#         self.use_res_scale = bool(use_res_scale)
#         self.res_scale = ResidualScale(C, init=1e-3) if self.use_res_scale else nn.Identity()
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         u = z.mean(dim=-1)  # [B,C,cyc]
#         cyc = u.size(-1)
#         if cyc <= 1:
#             return z
#         k_eff = max(3, min(self.k_cfg, 2 * cyc - 1))
#         rep = max(1, (k_eff // 3))
#         x = u
#         for _ in range(rep):
#             x = self.dw3(x)
#         x = self.pw(x); x = self.norm(x); x = self.act(x)
#         g = self.gate(u); x = x * g
#         x = x.unsqueeze(-1).expand_as(z)
#         x = self.res_scale(x) if self.use_res_scale else x
#         return z + x
#
# # ===== 带开关的 TimesBlock =====
# class AdvTimesBlock(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         # 基本超参
#         self.k        = configs.top_k
#         self.kernel   = configs.moving_avg
#         self.d_model  = configs.d_model
#         self.min_p    = getattr(configs, 'min_period', 3)
#         self.drop2d   = getattr(configs, 'conv2d_dropout', 0.0)
#
#         # 开关
#         self.use_series_decomp = bool(getattr(configs, 'use_series_decomp', 1))
#         self.use_sk            = bool(getattr(configs, 'use_sk', 1))
#         self.use_se            = bool(getattr(configs, 'use_se', 0))
#         self.se_strength       = float(getattr(configs, 'se_strength', 0.0))
#         self.use_cyc_conv1d    = bool(getattr(configs, 'use_cyc_conv1d', 1))
#         self.cyc_k             = int(getattr(configs, 'cyc_conv_kernel', 9))
#         self.use_gate_mlp      = bool(getattr(configs, 'use_gate_mlp', 1))
#         self.use_res_scale     = bool(getattr(configs, 'use_res_scale', 1))
#         self.reflect_pad       = bool(getattr(configs, 'reflect_pad', 1))
#         self.sk_tau            = float(getattr(configs, 'sk_tau', 1.5))
#
#         # 组件
#         self.decomp = SeriesDecomp(self.kernel) if self.use_series_decomp else None
#         self.block2d = Inception2dResBlockAdv(
#             self.d_model, use_sk=self.use_sk, use_se=self.use_se,
#             se_strength=self.se_strength, use_res_scale=self.use_res_scale,
#             sk_tau=self.sk_tau, drop=self.drop2d
#         )
#         self.cyc1d = (CycLargeKernelConv1dAdv(self.d_model, k=self.cyc_k, use_res_scale=self.use_res_scale)
#                       if self.use_cyc_conv1d else None)
#
#         hidden = max(8, self.d_model // 4)
#         self.gate_mlp = (nn.Sequential(nn.Linear(self.d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))
#                          if self.use_gate_mlp else None)
#         self.eps = 1e-8
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B,T,C=d_model]
#         """
#         B, T, C = x.shape
#         assert C == self.d_model, "Input last dim must equal d_model"
#
#         # 分解（可关）
#         if self.decomp is not None:
#             s, t = self.decomp(x)
#         else:
#             s, t = x, 0.0
#
#         # 周期候选
#         idx, w_prior = auto_correlation(s, self.k, min_p=self.min_p, max_p=T // 2)  # [B,k],[B,k]
#
#         agg_num = torch.zeros_like(x)
#         agg_den = x.new_zeros(B)
#
#         # 以 unique(p) 聚合，减少重复折叠
#         unique_p = torch.unique(idx)
#         for pv in unique_p.tolist():
#             mask = (idx == pv)
#             if not mask.any():
#                 continue
#             b_idx, j_idx = mask.nonzero(as_tuple=True)
#             m = b_idx.numel()
#
#             sb = s[b_idx]                                # [m,T,C]
#             wb = w_prior[b_idx, j_idx].view(m, 1, 1)     # [m,1,1]
#
#             # 折叠到 2D（可选反射填充）
#             z, _T = _fold_2d(sb, int(pv), reflect_pad=self.reflect_pad)  # [m,C,cyc,p]
#
#             # 多尺度 2D +（可选 SK/SE/残差缩放）
#             z = self.block2d(z)                          # [m,C,cyc,p]
#
#             # 可选：cyc 轴大核卷积
#             if self.cyc1d is not None:
#                 z = self.cyc1d(z)
#
#             # 展回 1D
#             y = _unfold_1d(z, _T)                        # [m,T,C]
#
#             # 融合权重：w_prior ×（可选 gate）
#             if self.gate_mlp is not None:
#                 gate = torch.sigmoid(self.gate_mlp(y.mean(dim=1)))  # [m,1]
#                 score = torch.log(wb + self.eps) + torch.log(gate.view(m, 1, 1) + self.eps)
#             else:
#                 score = torch.log(wb + self.eps)
#             score_exp = torch.exp(score)                 # [m,1,1]
#
#             contrib = y * score_exp                      # [m,T,C]
#             agg_num.index_add_(0, b_idx, contrib)
#             agg_den.index_add_(0, b_idx, score_exp.view(-1))
#
#         agg = agg_num / (agg_den.view(B, 1, 1) + self.eps)
#
#         # 加回趋势，并保留季节残差（若没分解则 t=0）
#         out = agg + (t if isinstance(t, torch.Tensor) else 0.0)
#         return out + (x - (t if isinstance(t, torch.Tensor) else 0.0))
#
# # ===== 继承 MyModule.Model，并替换 blocks =====
# class Model(MyModule.Model):
#     """
#     继承你的 MyModule.Model：
#     - super().__init__(configs) 后，直接替换 self.blocks 为 AdvTimesBlock 的 ModuleList
#     - 其它（project / norms / attention_pool / classifier）完全沿用父类实现
#     - 因为 forward 会遍历 self.blocks，所以无需重写 forward
#     """
#     def __init__(self, configs):
#         super().__init__(configs)
#         # 用带开关的 AdvTimesBlock 替换父类构建的 blocks
#         try:
#             self.blocks = nn.ModuleList([AdvTimesBlock(configs) for _ in range(configs.e_layers)])
#         except Exception as e:
#             print(f"[TimesNetAdv] WARNING: fallback to MyModule blocks due to: {e}")
#             # 出现异常则保留父类 blocks，不影响正常训练/推理
#             pass
