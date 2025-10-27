
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utils
# ----------------------------
def _best_gn_groups(C: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if C % g == 0:
            return g
    return 1


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x: [B, T, C], mask: [B, T] with 1 for valid, 0 for pad
    """
    if mask is None:
        return x.mean(dim=dim)
    w = mask.float().unsqueeze(-1)  # [B,T,1]
    s = (x * w).sum(dim=dim)
    d = w.sum(dim=dim).clamp_min(1e-6)
    return s / d


def _pick_period_top1(x: torch.Tensor, min_p: int = 3) -> torch.Tensor:
    """
    x: [B, T, C]
    return:
      periods: [B] (int64), 使用频谱幅值的 top-1
    """
    B, T, C = x.shape
    Xf = torch.fft.rfft(x, dim=1)          # [B, F, C]
    amp = (Xf.abs()).mean(dim=-1)          # [B, F]
    amp = amp[:, 1:]                       # 去直流分量
    # 若 T 很小，兜底
    if amp.size(1) == 0:
        p0 = max(min_p, min(T // 2, max(2, T // 4)))
        return torch.full((B,), max(p0, 3), dtype=torch.long, device=x.device)
    vals, idx = torch.topk(amp, k=1, dim=1)  # [B,1]
    f_idx = idx[:, 0] + 1                    # 还原真实频率索引
    periods = (T // f_idx).clamp(min=min_p, max=max(2, T // 2))
    return periods.long()


def _reshape_period(x: torch.Tensor, p: int) -> Tuple[torch.Tensor, int]:
    """
    x: [B,T,C] -> z: [B,C,cyc,p], and original T
    右侧使用反射填充到 p 的倍数
    """
    B, T, C = x.shape
    pad = ((T + p - 1) // p) * p - T
    if pad > 0:
        x_nct = x.permute(0, 2, 1).contiguous()      # [B,C,T]
        x_nct = F.pad(x_nct, (0, pad), mode='reflect')
        x = x_nct.permute(0, 2, 1).contiguous()      # [B,T+pad,C]
    T_new = x.size(1)
    cyc = T_new // p
    z = x.view(B, cyc, p, C).permute(0, 3, 1, 2).contiguous()  # [B,C,cyc,p]
    return z, T


def _unshape(z: torch.Tensor, T: int) -> torch.Tensor:
    """
    z: [B,C,cyc,p] -> [B,T,C] (truncate)
    """
    B, C, cyc, p = z.shape
    y = z.permute(0, 2, 3, 1).reshape(B, cyc * p, C)[:, :T, :].contiguous()
    return y


# ----------------------------
# 安全环绕填充（period 轴）
# ----------------------------
def _safe_circ_pad_p(z: torch.Tensor, pad: int) -> torch.Tensor:
    """
    z: [B,C,cyc,p]  ->  [B,C,cyc,p+2*pad]
    避免 torch 的 circular 超长环绕限制
    """
    if pad <= 0:
        return z
    p = z.size(-1)
    if pad <= p:
        left  = z[..., -pad:]
        right = z[..., :pad]
        return torch.cat([left, z, right], dim=-1)
    # pad > p：重复再切片
    reps = (pad + p - 1) // p
    z_rep = z.repeat(1, 1, 1, reps * 2 + 1)  # 中心 + 两侧
    center_start = reps * p
    left_start   = center_start - pad
    right_end    = center_start + p + pad
    return z_rep[..., left_start:right_end]


# ----------------------------
# 轻量 PeriodBlock（单分支 depthwise 1×k）
# ----------------------------
class LitePeriodBlock(nn.Module):
    """
    [B,C,cyc,p] → 沿 period 做 depthwise 1×k 卷积（k=5），GN+GELU
    没有注意力/没有多分支
    """
    def __init__(self, C: int, k: int = 5, drop2d: float = 0.0):
        super().__init__()
        g = _best_gn_groups(C)
        self.k = k
        self.pre = nn.GroupNorm(g, C)
        self.dw = nn.Conv2d(C, C, kernel_size=(1, k), padding=(0, 0), groups=C, bias=False)  # 我们手动做pad
        self.pw = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(g, C)
        self.act = nn.GELU()
        self.drop2d = nn.Dropout2d(drop2d) if drop2d > 0 else nn.Identity()
        # 轻量残差尺度
        self.gamma = nn.Parameter(torch.ones(1, C, 1, 1) * 1e-3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,C,cyc,p]
        x = self.pre(z)
        pad = self.k // 2
        x_pad = _safe_circ_pad_p(x, pad)     # 安全环绕填充
        y = self.dw(x_pad)
        y = self.pw(y)
        y = self.norm(y)
        y = self.act(y)
        return z + self.gamma * self.drop2d(y)


# ----------------------------
# Backbone：1~2 个 LitePeriodBlock（默认 1）
# ----------------------------
class LiteBackbone(nn.Module):
    def __init__(self, C: int, layers: int = 1, k: int = 5, drop2d: float = 0.0):
        super().__init__()
        layers = max(1, layers)
        self.blocks = nn.ModuleList([LitePeriodBlock(C, k=k, drop2d=drop2d) for _ in range(layers)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            z = blk(z)
        return z


# ----------------------------
# 弱化版 Model
# ----------------------------
class Model(nn.Module):
    """
    精简/弱化：固定 top-1 周期，无注意力、无 Inception
    - configs.enc_in: 输入特征维
    - configs.d_model: 隐层维（默认 64，更小更“弱化”）
    - configs.layers / configs.e_layers: LiteBlock 个数（默认 1）
    - configs.num_class: 类别数
    - configs.min_period: 最小周期（默认 3）
    - configs.conv2d_dropout: 2D dropout（默认 0.0）
    - configs.kernel_p: period 轴核大小（默认 5）
    """
    def __init__(self, configs):
        super().__init__()
        self.enc_in = getattr(configs, 'enc_in', 1)
        self.d_model = getattr(configs, 'd_model', 64)              # 更小的隐藏维度
        self.num_class = getattr(configs, 'num_class', 2)
        self.min_period = getattr(configs, 'min_period', 3)
        self.layers = getattr(configs, 'layers', getattr(configs, 'e_layers', 1))  # 默认 1 层
        self.drop2d = getattr(configs, 'conv2d_dropout', 0.0)
        self.kernel_p = getattr(configs, 'kernel_p', 5)              # 单一小核

        # 输入投影
        self.in_proj = nn.Linear(self.enc_in, self.d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_model)

        # 轻量骨干
        self.backbone = LiteBackbone(self.d_model, layers=self.layers, k=self.kernel_p, drop2d=self.drop2d)

        # 分类头
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.num_class)
        )

    def _period_path(self, x: torch.Tensor, p: int) -> torch.Tensor:
        """
        x: [B,T,C] -> [B,T,C] （C = d_model）
        """
        z, T = _reshape_period(x, p)          # [B,C,cyc,p]
        z = self.backbone(z)                   # 1~2 个轻量块
        y = _unshape(z, T)                     # [B,T,C]
        return y

    def forward(self, x, padding_mask=None, x_mark=None, dec_inp=None):
        """
        x: [B, L, enc_in]
        padding_mask: [B, L] (1=valid, 0=pad) or None
        """
        B, L, _ = x.shape
        x = self.in_proj(x)                    # [B,L,d_model]
        x = self.norm(x)

        # 仅取 top-1 周期
        periods = _pick_period_top1(x, min_p=self.min_period)  # [B]

        # 分组运行（相同 period 的样本一起算）
        fused = x.new_zeros(B, L, self.d_model)
        uniq = torch.unique(periods)
        for pval in uniq.tolist():
            sel = (periods == pval)
            if sel.any():
                xi = x[sel]                             # [b_i, L, C]
                yi = self._period_path(xi, int(pval))  # [b_i, L, C]
                fused[sel] = yi

        # 时间维 masked mean -> 分类
        feat = _masked_mean(fused, padding_mask, dim=1)  # [B,C]
        logits = self.head(feat)                         # [B,num_class]
        return logits
