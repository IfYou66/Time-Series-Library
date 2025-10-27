# models/OSCNN_Lite.py  （你也可以命名为 OSCNN.py / MyModule.py）
# Omni-Scale CNN (Lite) for Multivariate Time-Series Classification
# 低容量、强正则、抗过拟合版本

from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 简易 DropPath（Stochastic Depth）
# ----------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # [B,1,1] or [B,1,1] 按 batch 维广播
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


# ----------------------------
# 掩码平均池化（沿时间维）
# ----------------------------
def masked_mean_1d(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x:    [B, T, C]
    mask: [B, T]，1=有效，0=padding
    """
    if mask is None:
        return x.mean(dim=1)
    w = mask.float().unsqueeze(-1)  # [B,T,1]
    s = (x * w).sum(dim=1)          # [B,C]
    d = w.sum(dim=1).clamp_min(eps) # [B,1]
    return s / d


# ----------------------------
# 深度可分离卷积
# ----------------------------
class SepConv1d(nn.Module):
    def __init__(self, ch: int, k: int, padding: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=k, padding=padding, groups=ch, bias=False),
            nn.Conv1d(ch, ch, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# 弱化版 OmniScaleBlock
#   - 少分支（默认 kernels=[3,7]）
#   - GroupNorm + GELU
#   - 通道Dropout（Dropout2d）+ 时间Dropout（Dropout）
#   - DropPath 残差
# ----------------------------
class OmniScaleBlockLite(nn.Module):
    def __init__(
        self,
        channels: int,
        kernels: Iterable[int] = (3, 7),
        gn_groups: int = 8,
        drop_channel: float = 0.1,   # 通道级丢弃（Dropout2d）
        drop_time: float = 0.1,      # 时间级丢弃（Dropout）
        drop_path: float = 0.1       # 残差级随机深度
    ):
        super().__init__()
        self.pre = nn.GroupNorm(gn_groups, channels)

        # 多尺度分支（极简）
        ks = list(kernels)
        self.branches = nn.ModuleList()
        for k in ks:
            pad = k // 2
            self.branches.append(
                nn.Sequential(
                    SepConv1d(channels, k=k, padding=pad),
                    nn.GroupNorm(gn_groups, channels),
                    nn.GELU()
                )
            )

        # 1x1 融合 + 轻量 refine
        self.fuse = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, channels),
            nn.GELU()
        )
        self.refine = nn.Sequential(
            SepConv1d(channels, k=3, padding=1),
            nn.GroupNorm(gn_groups, channels),
            nn.GELU()
        )

        # 正则化
        self.drop_channel = nn.Dropout2d(drop_channel) if drop_channel > 0 else nn.Identity()
        self.drop_time = nn.Dropout(drop_time) if drop_time > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # 残差缩放
        self.gamma = nn.Parameter(torch.ones(1, channels, 1) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        z = self.pre(x)
        outs = [b(z) for b in self.branches]
        y = sum(outs) / len(outs)           # 均值聚合
        y = self.fuse(y)
        y = self.refine(y)

        # 通道/时间正则（注意：Dropout2d 作用于 [B,C,T] 会整通道置零）
        y = self.drop_channel(y)
        y = self.drop_time(y)

        # 残差 + DropPath
        y = self.gamma * y
        return x + self.drop_path(y)


# ----------------------------
# 主干：2 个（默认）Lite OS 块
# ----------------------------
class OSCNNBackboneLite(nn.Module):
    def __init__(
        self,
        channels: int,
        n_blocks: int = 2,
        kernels: Iterable[int] = (3, 7),
        gn_groups: int = 8,
        drop_channel: float = 0.1,
        drop_time: float = 0.1,
        drop_path: float = 0.1
    ):
        super().__init__()
        n_blocks = max(1, n_blocks)
        self.blocks = nn.ModuleList([
            OmniScaleBlockLite(
                channels=channels,
                kernels=kernels,
                gn_groups=gn_groups,
                drop_channel=drop_channel,
                drop_time=drop_time,
                drop_path=drop_path
            ) for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# ----------------------------
# 入口模型：与 Exp_Classification 对齐
# ----------------------------
class Model(nn.Module):
    """
    建议默认（更弱化）：
      - d_model: 64（可设 32）
      - e_layers/layers: 2
      - os_kernels: [3,7]
      - gn_groups: 8
      - drop_channel: 0.1, drop_time: 0.1, drop_path: 0.1
    其它可从 args 里读到并覆盖。
    """
    def __init__(self, configs):
        super().__init__()
        self.enc_in = getattr(configs, 'enc_in', 1)
        self.d_model = getattr(configs, 'd_model', 64)              # 更小通道
        self.num_class = getattr(configs, 'num_class', 2)

        n_blocks = getattr(configs, 'e_layers', getattr(configs, 'layers', 2))
        kernels = tuple(getattr(configs, 'os_kernels', [3, 7]))
        gn_groups = getattr(configs, 'gn_groups', 8)

        drop_channel = getattr(configs, 'drop_channel', 0.1)
        drop_time = getattr(configs, 'drop_time', 0.1)
        drop_path = getattr(configs, 'drop_path', 0.1)

        # 输入映射：[B,L,enc_in] -> [B,C,T]
        self.in_proj = nn.Linear(self.enc_in, self.d_model, bias=False)
        self.in_norm = nn.GroupNorm(gn_groups, self.d_model)

        # 主干（Lite）
        self.backbone = OSCNNBackboneLite(
            channels=self.d_model,
            n_blocks=n_blocks,
            kernels=kernels,
            gn_groups=gn_groups,
            drop_channel=drop_channel,
            drop_time=drop_time,
            drop_path=drop_path
        )

        # 分类头（再加正则）
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(0.2),
            nn.Linear(self.d_model, self.num_class)
        )

    def forward(self, x, padding_mask=None, x_mark=None, dec_inp=None):
        """
        x: [B, L, enc_in]
        padding_mask: [B, L] (1=valid, 0=pad)
        """
        B, L, _ = x.shape
        x = self.in_proj(x)                    # [B,L,C]
        x = x.transpose(1, 2).contiguous()     # [B,C,L]
        x = self.in_norm(x)                    # GN 稳定小 batch
        feat = self.backbone(x)                # [B,C,T]
        feat_t = feat.transpose(1, 2).contiguous()  # [B,T,C]
        pooled = masked_mean_1d(feat_t, padding_mask)  # [B,C]
        logits = self.head(pooled)             # [B,num_class]
        return logits
