import math
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
    Pm = P.mean(dim=-1)                            # [B,F]
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
        self.tau = tau                                # softmax 温度

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
            if w_prior.dim() == 4:                      # [B,1,1,1]
                w_prior = w_prior.view(B, 1, 1, 1, 1).expand(B, self.n, C, 1, 1)
            elif w_prior.dim() == 5 and w_prior.size(1) == 1:  # [B,1,1,1,1]
                w_prior = w_prior.expand(B, self.n, C, 1, 1)
            a = a + self.alpha * w_prior
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
    def __init__(self, channels: int, k: int = 9):
        super().__init__()
        C = channels
        self.k_cfg = k
        g = _best_gn_groups(C)
        # 用 3x depthwise 堆叠近似大核，autotune 更稳
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
# 7.5) 位置编码（正弦）用于 cyc 轴 Transformer
# =========================
class SinusoidalPosEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]  (batch_first)
        """
        B, L, C = x.shape
        device = x.device
        pe = torch.zeros(L, C, device=device)
        position = torch.arange(0, L, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, C, 2, device=device) * (-math.log(10000.0) / C))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).expand(B, L, C)
        return x + pe

# =========================
# 7.6) cyc 轴 Transformer（轴向注意力）+ 残差缩放
# =========================
class CycTransformerBlock(nn.Module):
    """
    在 [B,C,cyc,p] 的特征上，先对 p 轴做聚合得到 [B,cyc,C]，
    仅沿 cyc 轴堆叠 TransformerEncoderLayer，然后回广播到 2D 并残差融合。
    """
    def __init__(
        self,
        channels: int,
        nhead: int = 4,
        num_layers: int = 1,
        ffn_scale: float = 4.0,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
    ):
        super().__init__()
        d_model = channels
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(ffn_scale * d_model),
            dropout=dropout,
            activation='gelu',
            batch_first=True,            # [B, L, C]
            norm_first=True              # Pre-Norm，稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.posenc = SinusoidalPosEncoding(d_model) if use_pos_encoding else nn.Identity()
        self.res_scale = ResidualScale(channels, init=1e-3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,C,cyc,p]
        u = z.mean(dim=-1)                 # [B,C,cyc]
        u = u.transpose(1, 2).contiguous() # [B,cyc,C]
        if u.size(1) <= 1:                 # 边界保护
            return z
        u = self.posenc(u)                 # 位置编码
        y = self.encoder(u)                # [B,cyc,C]
        y = y.transpose(1, 2).unsqueeze(-1).expand_as(z)  # [B,C,cyc,p]
        return z + self.res_scale(y)


# =========================
# 8) TimesBlock（集成：SK + cyc-Conv1d 或 cyc-Transformer + SE）
# =========================
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.k        = configs.top_k
        self.kernel   = configs.moving_avg
        self.d_model  = configs.d_model
        self.min_p    = getattr(configs, 'min_period', 3)
        self.drop2d   = getattr(configs, 'conv2d_dropout', 0.0)

        # —— 三件套参数 —— #
        # （1）SK 温度与 SE 强度
        self.sk_tau   = getattr(configs, 'sk_tau', 1.5)          # 初期更均匀
        self.se_strength = getattr(configs, 'se_strength', 0.0)  # 先关/弱开
        # （2）cyc 轴分支：Conv1d or Transformer
        self.use_cyc_transformer = getattr(configs, 'use_cyc_transformer', False)
        self.cyc_conv_kernel     = getattr(configs, 'cyc_conv_kernel', 9)   # Conv1d 用
        self.cyc_nhead           = getattr(configs, 'cyc_nhead', 4)
        self.cyc_layers          = getattr(configs, 'cyc_layers', 1)
        self.cyc_ffn_scale       = getattr(configs, 'cyc_ffn_scale', 4.0)
        self.cyc_dropout         = getattr(configs, 'cyc_dropout', 0.1)
        self.cyc_posenc          = getattr(configs, 'cyc_posenc', True)

        self.decomp   = SeriesDecomp(self.kernel)
        self.block2d  = Inception2dResBlock_SKSE(
            self.d_model, drop=self.drop2d, sk_tau=self.sk_tau, se_strength=self.se_strength
        )
        if self.use_cyc_transformer:
            self.cyc_block = CycTransformerBlock(
                channels=self.d_model,
                nhead=self.cyc_nhead,
                num_layers=self.cyc_layers,
                ffn_scale=self.cyc_ffn_scale,
                dropout=self.cyc_dropout,
                use_pos_encoding=self.cyc_posenc,
            )
        else:
            self.cyc_block = CycLargeKernelConv1d(self.d_model, k=self.cyc_conv_kernel)

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

            # cyc 轴分支（Conv1d 或 Transformer）
            z = self.cyc_block(z)                        # [m,C,cyc,p]

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

        # 初始化更稳一些
        nn.init.trunc_normal_(self.pool_q, std=0.02)

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

# ------------------------------------------------------------
# version4 beat
# 加入环形对齐模块 CircularConvAlongP（在周期维 p 上做 depthwise 1×k 的 circular padding 卷积）→ 提升对相位漂移的鲁棒性而无需显式 roll。
# 将 Conv2dResBlock 升级为轻量 Inception 多尺度块（3×3、5×5、dilated‑3×3 的并行深度可分离卷积 + 1×1 融合 + 残差），并用 GroupNorm 提升小批次稳定性。
# 高效分支聚合：按 unique(p) 进行分桶，对每个 p 一次性处理所有 (b, j) 选择，用 index_add_ 实现对 agg_num/agg_den 的张量化累加，去掉内层 j 循环。
# 反射填充改为通道优先安全实现（[B,C,T] 上 ReflectionPad1d），并对 auto_correlation 做了无效候选回退，避免 NaN。
# ------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
#
# # =========================
# # 工具：安全的 GroupNorm 组数选择（保证可整除）
# # =========================
# def _best_gn_groups(C: int) -> int:
#     for g in [32, 16, 8, 4, 2, 1]:
#         if C % g == 0:
#             return g
#     return 1
#
# # =========================
# # 残差缩放（LayerScale 风格）
# # =========================
# class ResidualScale(nn.Module):
#     def __init__(self, channels: int, init: float = 1e-3):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * init)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * self.gamma
#
# # =========================
# # 1) 序列分解
# # =========================
# class SeriesDecomp(nn.Module):
#     def __init__(self, kernel_size: int):
#         super().__init__()
#         self.avg_pool = nn.AvgPool1d(
#             kernel_size=kernel_size, stride=1,
#             padding=(kernel_size - 1) // 2, count_include_pad=False
#         )
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # x: [B,T,C]
#         x_nct = x.permute(0, 2, 1)                     # [B,C,T]
#         trend = self.avg_pool(x_nct).permute(0, 2, 1)  # [B,T,C]
#         seasonal = x - trend
#         return seasonal, trend
#
#
# # =========================
# # 2) 自相关（稳健回退）
# # =========================
# def auto_correlation(x: torch.Tensor, k: int, min_p: int = 3, max_p: int = None):
#     """
#     x: [B, T, C]
#     return:
#       idx: [B, k]   每个样本的周期滞后
#       w  : [B, k]   自相关峰值 softmax 权重（先验）
#     """
#     B, T, C = x.shape
#     dtype = x.dtype
#     if max_p is None:
#         max_p = max(1, T // 2)
#     max_p = max(min_p, min(max_p, T - 1))
#
#     Xf = torch.fft.rfft(x, dim=1)                  # [B,F,C]
#     P = (Xf * torch.conj(Xf)).real                 # [B,F,C]
#     Pm = P.mean(dim=-1)                            # [B,F]（如需更强可做可学习通道加权）
#     r = torch.fft.irfft(Pm, n=T, dim=1)            # [B,T]
#
#     # 合法掩码
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
#     # 若全非法，回退到 p0
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
# # =========================
# # 3) 工具：安全反射折叠/展开
# # =========================
# def _fold_2d_reflect(x_1d: torch.Tensor, p: int) -> Tuple[torch.Tensor, int]:
#     """
#     x_1d: [b,T,C] → z: [b,C,cyc,p], T_orig
#     """
#     b, T, C = x_1d.shape
#     pad = ((T + p - 1) // p) * p - T
#     if pad > 0:
#         x_nct = x_1d.permute(0, 2, 1).contiguous()     # [b,C,T]
#         x_nct = F.pad(x_nct, (0, pad), mode='reflect') # 仅右侧填充
#         x_1d  = x_nct.permute(0, 2, 1).contiguous()    # [b,T+pad,C]
#     T_new = x_1d.shape[1]
#     cyc = T_new // p
#     z = x_1d.reshape(b, cyc, p, C).permute(0, 3, 1, 2).contiguous()  # [b,C,cyc,p]
#     return z, T
#
#
# def _unfold_1d(z_2d: torch.Tensor, T: int) -> torch.Tensor:
#     """
#     z_2d: [b,C,cyc,p] → y: [b,T,C]
#     """
#     b, C, cyc, p = z_2d.shape
#     y = z_2d.permute(0, 2, 3, 1).reshape(b, cyc * p, C)[:, :T, :].contiguous()
#     return y
#
#
# # =========================
# # 4) SE（通道注意力）
# # =========================
# class SEBlock(nn.Module):
#     def __init__(self, channels: int, reduction: int = 8):
#         super().__init__()
#         hidden = max(reduction, channels // 4)
#         g = _best_gn_groups(channels)
#         # 用 Conv2d 实现 SE，保持轻量
#         self.fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),              # [B,C,1,1]
#             nn.Conv2d(channels, hidden, 1, bias=False), nn.GELU(),
#             nn.Conv2d(hidden, channels, 1, bias=False), nn.Sigmoid()
#         )
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         # z: [B,C,cyc,p]
#         w = self.fc(z)
#         return z * w
#
#
# # =========================
# # 5) SK 融合（支持温度/先验）
# # =========================
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
#         self.alpha = nn.Parameter(torch.tensor(0.5))  # 频谱先验权重（如需）
#         self.tau = tau                                # softmax 温度：>1 更均匀，<1 更尖锐
#
#     def forward(self, feats, w_prior: torch.Tensor = None):
#         """
#         feats: List of [B,C,cyc,p], len = n
#         w_prior: None 或 [B,1,1,1] / [B, n, 1, 1] / [B,n,C,1,1]
#         """
#         B, C, _, _ = feats[0].shape
#         s = sum(feats) / self.n                         # [B,C,cyc,p]
#         a = self.squeeze(s).view(B, self.n, C, 1, 1)    # [B,n,C,1,1]
#         if w_prior is not None:
#             # 广播到 [B,n,C,1,1]
#             if w_prior.dim() == 4:      # [B,1,1,1]
#                 w_prior = w_prior.view(B, 1, 1, 1, 1).expand(B, self.n, C, 1, 1)
#             elif w_prior.dim() == 5 and w_prior.size(1) == 1:  # [B,1,1,1,1]
#                 w_prior = w_prior.expand(B, self.n, C, 1, 1)
#             a = a + self.alpha * w_prior
#         # 温度软化
#         w = torch.softmax(a / self.tau, dim=1)          # [B,n,C,1,1]
#         out = sum(w[:, i] * feats[i] for i in range(self.n))   # [B,C,cyc,p]
#         return out
#
#
# # =========================
# # 6) Inception 2D 残差块（多尺度 + SK + SE）+ Pre-Norm + 残差缩放
# # =========================
# class Inception2dResBlock_SKSE(nn.Module):
#     def __init__(self, channels: int, drop: float = 0.0, sk_tau: float = 1.5, se_strength: float = 0.0):
#         super().__init__()
#         C = channels
#         g = _best_gn_groups(C)
#
#         # Pre-Norm
#         self.pre = nn.GroupNorm(num_groups=g, num_channels=C)
#
#         # 三个深度可分离分支
#         self.b3   = nn.Sequential(
#             nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False),
#             nn.GroupNorm(num_groups=g, num_channels=C), nn.GELU()
#         )
#         self.b5   = nn.Sequential(
#             nn.Conv2d(C, C, kernel_size=5, padding=2, groups=C, bias=False),
#             nn.GroupNorm(num_groups=g, num_channels=C), nn.GELU()
#         )
#         self.bDil = nn.Sequential(
#             nn.Conv2d(C, C, kernel_size=3, padding=2, dilation=2, groups=C, bias=False),
#             nn.GroupNorm(num_groups=g, num_channels=C), nn.GELU()
#         )
#         self.sk   = SKMerge(C, n_branches=3, reduction=max(8, C // 4), tau=sk_tau)
#         self.merge_pw = nn.Conv2d(C, C, kernel_size=1, bias=False)
#         self.merge_gn = nn.GroupNorm(num_groups=g, num_channels=C)
#         self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
#
#         # 第二阶段轻量卷积
#         self.dw   = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
#         self.pw   = nn.Conv2d(C, C, kernel_size=1, bias=False)
#         self.gn2  = nn.GroupNorm(num_groups=g, num_channels=C)
#         self.act  = nn.GELU()
#
#         # 通道 SE 门控（强度可控）
#         self.se = SEBlock(C, reduction=max(8, C // 4))
#         self.se_strength = se_strength
#
#         # 残差缩放
#         self.res_scale = ResidualScale(C, init=1e-3)
#
#     def forward(self, z: torch.Tensor, w_prior_branch: torch.Tensor = None) -> torch.Tensor:
#         # z: [B,C,cyc,p]
#         x_in = self.pre(z)                                # Pre-Norm
#         f1, f2, f3 = self.b3(x_in), self.b5(x_in), self.bDil(x_in)
#         x = self.sk([f1, f2, f3], w_prior=w_prior_branch) # [B,C,cyc,p]
#         x = self.merge_pw(x)
#         x = self.merge_gn(x)
#         x = self.act(x)
#         x = self.drop(x)
#
#         x = self.dw(x)
#         x = self.pw(x)
#         x = self.gn2(x)
#         x = self.act(x)
#
#         if self.se_strength > 0:
#             # 将 SE 当成缓和因子：pow(se_strength)
#             x = x * (self.se(x).pow(self.se_strength))
#
#         return z + self.res_scale(x)                      # 缩放后的残差
#
#
# # =========================
# # 7) cyc 轴大核 Conv1d（动态有效核长 + 3x堆叠近似 + 轻量门控 + 残差缩放）
# # =========================
# class CycLargeKernelConv1d(nn.Module):
#     """
#     在 [B,C,cyc,p] 的特征上，先对 p 轴做聚合得到 [B,C,cyc]，
#     然后沿 cyc 轴做深度可分离 1D 大核卷积（用多次3x卷积近似），再广播回 2D。
#     """
#     def __init__(self, channels: int, k: int = 15):
#         super().__init__()
#         C = channels
#         self.k_cfg = k
#         g = _best_gn_groups(C)
#         # 用 3x depthwise 近似大核，autotune 更稳
#         self.dw3 = nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C, bias=False)
#         self.pw  = nn.Conv1d(C, C, kernel_size=1, bias=False)
#         self.norm = nn.GroupNorm(num_groups=g, num_channels=C)
#         self.act  = nn.GELU()
#         # 轻量门控，避免覆盖原特征
#         self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(C, C, 1, bias=False), nn.Sigmoid())
#         # 残差缩放
#         self.res_scale = ResidualScale(C, init=1e-3)
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         # z: [B,C,cyc,p]
#         u = z.mean(dim=-1)                               # [B,C,cyc]
#         cyc = u.size(-1)
#         if cyc <= 1:
#             return z  # 边界保护
#
#         # 有效核长不超过 2*cyc-1，且不小于3
#         k_eff = max(3, min(self.k_cfg, 2 * cyc - 1))
#         rep = max(1, (k_eff // 3))
#
#         x = u
#         for _ in range(rep):
#             x = self.dw3(x)
#         x = self.pw(x)
#         x = self.norm(x)
#         x = self.act(x)
#
#         g = self.gate(u)                                 # [B,C,1]
#         x = x * g                                        # 轻量门控
#         x = x.unsqueeze(-1).expand_as(z)                 # [B,C,cyc,p]
#         return z + self.res_scale(x)                     # 缩放后的残差
#
#
# # =========================
# # 8) TimesBlock（集成：SK + cyc-Conv1d + SE）
# # =========================
# class TimesBlock(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.k        = configs.top_k
#         self.kernel   = configs.moving_avg
#         self.d_model  = configs.d_model
#         self.min_p    = getattr(configs, 'min_period', 3)
#         self.drop2d   = getattr(configs, 'conv2d_dropout', 0.0)
#         self.cyc_k    = getattr(configs, 'cyc_conv_kernel', 9)   # 起步用 9 更稳
#         self.sk_tau   = getattr(configs, 'sk_tau', 1.5)          # 初期更均匀
#         self.se_strength = getattr(configs, 'se_strength', 0.0)  # 先关/弱开
#
#         self.decomp   = SeriesDecomp(self.kernel)
#         self.block2d  = Inception2dResBlock_SKSE(self.d_model, drop=self.drop2d,
#                                                  sk_tau=self.sk_tau, se_strength=self.se_strength)
#         self.cyc1d    = CycLargeKernelConv1d(self.d_model, k=self.cyc_k)
#
#         # 内容感知门控（样本级标量，用于最终融合）
#         hidden = max(8, self.d_model // 4)
#         self.gate_mlp = nn.Sequential(
#             nn.Linear(self.d_model, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, 1)
#         )
#         self.eps = 1e-8
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B,T,C=d_model]
#         """
#         B, T, C = x.shape
#         assert C == self.d_model, "Input last dim must equal d_model"
#
#         # 分解
#         s, t = self.decomp(x)  # [B,T,C]
#
#         # 候选周期与先验
#         idx, w_prior = auto_correlation(s, self.k, min_p=self.min_p, max_p=T // 2)  # [B,k],[B,k]
#
#         # 聚合容器
#         agg_num = torch.zeros_like(x)    # [B,T,C]
#         agg_den = x.new_zeros(B)         # [B]
#
#         # 以 unique(p) 分桶
#         unique_p = torch.unique(idx)
#         for pv in unique_p.tolist():
#             mask = (idx == pv)                          # [B,k]
#             if not mask.any():
#                 continue
#             b_idx, j_idx = mask.nonzero(as_tuple=True)  # [m],[m]
#             m = b_idx.numel()
#
#             sb = s[b_idx]                                # [m,T,C]
#             wb = w_prior[b_idx, j_idx].view(m, 1, 1)     # [m,1,1]
#
#             # 折叠
#             z, _T = _fold_2d_reflect(sb, int(pv))        # [m,C,cyc,p]
#
#             # 2D 多尺度 + SK + SE（含 Pre-Norm/残差缩放）
#             z = self.block2d(z)                          # [m,C,cyc,p]
#
#             # cyc 轴大核 Conv1d（跨周期长程，含残差缩放/门控/动态有效核长）
#             z = self.cyc1d(z)                            # [m,C,cyc,p]
#
#             # 展回 1D
#             y = _unfold_1d(z, _T)                        # [m,T,C]
#
#             # 样本级内容门控
#             gate = torch.sigmoid(self.gate_mlp(y.mean(dim=1)))   # [m,1]
#             score = torch.log(wb + self.eps) + torch.log(gate.view(m, 1, 1) + self.eps)
#             score_exp = torch.exp(score)                 # [m,1,1]
#
#             contrib = y * score_exp                      # [m,T,C]
#             agg_num.index_add_(0, b_idx, contrib)
#             agg_den.index_add_(0, b_idx, score_exp.view(-1))
#
#         # 归一化融合
#         agg = agg_num / (agg_den.view(B, 1, 1) + self.eps)
#
#         # 加回趋势，并保留季节残差
#         out = agg + t
#         return out + (x - t)
#
#
# # =========================
# # 9) 分类模型（与原接口一致）
# # =========================
# class Model(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.d_model   = configs.d_model
#         self.num_class = configs.num_class
#         self.dropout_p = configs.dropout
#
#         self.project = nn.Linear(configs.enc_in, self.d_model)
#         self.blocks  = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
#         self.norms   = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(configs.e_layers)])
#
#         # 注意力池化（可学习 query）
#         self.pool_q = nn.Parameter(torch.randn(1, 1, self.d_model))
#         self.dropout    = nn.Dropout(self.dropout_p)
#         self.classifier = nn.Linear(self.d_model, self.num_class)
#
#     def attention_pool(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B,T,d] → [B,d]
#         B, T, d = x.shape
#         q = self.pool_q.expand(B, -1, -1)                       # [B,1,d]
#         att = torch.matmul(q, x.transpose(1, 2)) / (d ** 0.5)   # [B,1,T]
#         att = torch.softmax(att, dim=-1)
#         return torch.matmul(att, x).squeeze(1)                  # [B,d]
#
#     def forward(self, x_enc: torch.Tensor, *args) -> torch.Tensor:
#         x = self.project(x_enc)                                  # [B,T,d]
#         for blk, ln in zip(self.blocks, self.norms):
#             x = ln(blk(x))                                       # [B,T,d]
#         x = self.attention_pool(x)                               # [B,d]
#         x = self.dropout(x)
#         out = self.classifier(x)                                 # [B,num_class]
#         return out


# ------------------------------------------------------------
# version3
# 加入环形对齐模块 CircularConvAlongP（在周期维 p 上做 depthwise 1×k 的 circular padding 卷积）→ 提升对相位漂移的鲁棒性而无需显式 roll。
# 将 Conv2dResBlock 升级为轻量 Inception 多尺度块（3×3、5×5、dilated‑3×3 的并行深度可分离卷积 + 1×1 融合 + 残差），并用 GroupNorm 提升小批次稳定性。
# 高效分支聚合：按 unique(p) 进行分桶，对每个 p 一次性处理所有 (b, j) 选择，用 index_add_ 实现对 agg_num/agg_den 的张量化累加，去掉内层 j 循环。
# 反射填充改为通道优先安全实现（[B,C,T] 上 ReflectionPad1d），并对 auto_correlation 做了无效候选回退，避免 NaN。
# ------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
#
# # ------------------------------------------------------------
# # 1) 序列分解：SeriesDecomp（中心移动平均）
# #    采用 [B,C,T] 上的 AvgPool1d，避免维度歧义
# # ------------------------------------------------------------
# class SeriesDecomp(nn.Module):
#     def __init__(self, kernel_size: int):
#         super().__init__()
#         self.avg_pool = nn.AvgPool1d(
#             kernel_size=kernel_size, stride=1,
#             padding=(kernel_size - 1) // 2, count_include_pad=False
#         )
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         x: [B, T, C]
#         return: seasonal, trend
#         """
#         x_nct = x.permute(0, 2, 1)                  # [B,C,T]
#         trend = self.avg_pool(x_nct).permute(0, 2, 1)  # [B,T,C]
#         seasonal = x - trend
#         return seasonal, trend
#
#
# # ------------------------------------------------------------
# # 2) 自相关（逐样本）+ 健壮性回退
# #    - 限制候选周期范围
# #    - 避免全 -inf / NaN：回退到默认 p0 并给出均匀权重
# # ------------------------------------------------------------
# def auto_correlation(x: torch.Tensor, k: int, min_p: int = 3, max_p: int = None):
#     """
#     x: [B, T, C]
#     return:
#       idx: [B, k]   每个样本的周期(滞后)
#       w  : [B, k]   对应 softmax 权重（来自自相关峰值，仅作先验）
#     """
#     B, T, C = x.shape
#     device = x.device
#     dtype = x.dtype
#     if max_p is None:
#         max_p = max(1, T // 2)
#     max_p = max(min_p, min(max_p, T - 1))  # 保证范围合法
#
#     # 频域功率谱（逐样本）
#     Xf = torch.fft.rfft(x, dim=1)                      # [B, F, C]
#     P = (Xf * torch.conj(Xf)).real                     # [B, F, C]
#     # 通道均值聚合（如需更强可改为可学习通道权重）
#     Pm = P.mean(dim=-1)                                # [B, F]
#     r = torch.fft.irfft(Pm, n=T, dim=1)                # [B, T]
#
#     # 屏蔽非法滞后
#     r_mask = torch.ones_like(r, dtype=torch.bool)      # True 表示可选
#     r_mask[:, 0] = False
#     if min_p > 1:
#         r_mask[:, 1:min_p] = False
#     if max_p + 1 < T:
#         r_mask[:, max_p + 1:] = False
#
#     # 用一个非常小的值代替非法位置，避免 -inf 带来的 NaN
#     very_neg = torch.finfo(dtype).min / 8 if dtype.is_floating_point else -1e9
#     r_masked = torch.where(r_mask, r, torch.full_like(r, very_neg))
#
#     # 如果该样本全部非法，则回退到 p0 = clip(T//4, [min_p, max_p])
#     p0 = max(min_p, min(max_p, max(2, T // 4)))
#     all_bad = (r_mask.sum(dim=1) == 0)
#     if all_bad.any():
#         r_masked[all_bad] = very_neg
#         r_masked[all_bad, p0] = 0.0  # 人为放一个峰值，保证 topk 可用
#
#     k_eff = min(k, max_p - min_p + 1)  # 有效最大候选数
#     vals, idx = torch.topk(r_masked, k=k_eff, dim=1)   # [B,k_eff]
#     # 如果 k_eff < k，补齐（复制最后一个）
#     if k_eff < k:
#         pad_n = k - k_eff
#         idx = torch.cat([idx, idx[:, -1:].expand(B, pad_n)], dim=1)
#         vals = torch.cat([vals, vals[:, -1:].expand(B, pad_n)], dim=1)
#
#     # softmax 权重（先验）
#     w = F.softmax(vals, dim=1)
#     return idx, w
#
#
# # ------------------------------------------------------------
# # 3) 相位鲁棒：周期维上的环形卷积（depthwise 1×k + circular padding）
# # ------------------------------------------------------------
# class CircularConvAlongP(nn.Module):
#     def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
#         super().__init__()
#         assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
#         self.pad = (kernel_size // 2) * dilation
#         self.dw = nn.Conv2d(
#             channels, channels,
#             kernel_size=(1, kernel_size),
#             dilation=(1, dilation),
#             groups=channels,
#             bias=False,
#             padding=0,
#         )
#         self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
#         self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
#         self.act = nn.GELU()
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         """
#         z: [B, C, cyc, p]
#         """
#         if self.pad > 0:
#             # 只在最后一维 p 上做环形填充：F.pad 的顺序是 (W_left, W_right, H_top, H_bottom)
#             z_pad = F.pad(z, (self.pad, self.pad, 0, 0), mode='circular')
#         else:
#             z_pad = z
#         out = self.dw(z_pad)
#         out = self.pw(out)
#         out = self.norm(out)
#         out = self.act(out)
#         return z + out  # 残差
#
#
# # ------------------------------------------------------------
# # 4) 轻量 Inception 多尺度 2D 残差块（深度可分离卷积 + 1×1 融合）
# # ------------------------------------------------------------
# class Inception2dResBlock(nn.Module):
#     def __init__(self, channels: int, drop: float = 0.0):
#         super().__init__()
#         # 三个并行分支：3x3，5x5，膨胀3x3(dil=2)
#         self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
#         self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
#         self.dwd = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False)
#
#         self.pw_merge = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
#
#         self.norm1 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
#         self.act1  = nn.GELU()
#         self.drop  = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
#
#         # 第二阶段再做一轮轻量卷积 + 融合
#         self.dw3_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
#         self.pw2   = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
#         self.norm2 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         """
#         z: [B, C, cyc, p]
#         """
#         b1 = self.dw3(z)
#         b2 = self.dw5(z)
#         b3 = self.dwd(z)
#         x  = torch.cat([b1, b2, b3], dim=1)
#         x  = self.pw_merge(x)
#         x  = self.norm1(x)
#         x  = self.act1(x)
#         x  = self.drop(x)
#
#         x  = self.dw3_2(x)
#         x  = self.pw2(x)
#         x  = self.norm2(x)
#         return z + x  # 残差
#
#
# # ------------------------------------------------------------
# # 5) 安全反射填充 + 折叠/展开
# #    只在时间维 T 上反射填充（在 [B,C,T] 形态）
# # ------------------------------------------------------------
# def _fold_2d_reflect(x_1d: torch.Tensor, p: int) -> Tuple[torch.Tensor, int]:
#     """
#     x_1d: [b, T, C]
#     p:    period
#     return z: [b, C, cyc, p], T_orig
#     """
#     b, T, C = x_1d.shape
#     pad = ((T + p - 1) // p) * p - T
#     if pad > 0:
#         x_nct = x_1d.permute(0, 2, 1).contiguous()     # [b,C,T]
#         # 仅右侧填充 pad；Reflection 要求 pad <= T-1，且我们有 p <= T//2，安全
#         x_nct = F.pad(x_nct, (0, pad), mode='reflect')
#         x_1d  = x_nct.permute(0, 2, 1).contiguous()    # [b,T+pad,C]
#     T_new = x_1d.shape[1]
#     cyc = T_new // p
#     z = x_1d.reshape(b, cyc, p, C).permute(0, 3, 1, 2).contiguous()  # [b,C,cyc,p]
#     return z, T
#
#
# def _unfold_1d(z_2d: torch.Tensor, T: int) -> torch.Tensor:
#     """
#     z_2d: [b, C, cyc, p]
#     return y: [b, T, C]
#     """
#     b, C, cyc, p = z_2d.shape
#     y = z_2d.permute(0, 2, 3, 1).reshape(b, cyc * p, C)[:, :T, :].contiguous()  # [b,T,C]
#     return y
#
#
# # ------------------------------------------------------------
# # 6) TimesBlock：相位鲁棒 + 多尺度 + 高效分支聚合
# # ------------------------------------------------------------
# class TimesBlock(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.k        = configs.top_k
#         self.kernel   = configs.moving_avg
#         self.d_model  = configs.d_model
#         self.min_p    = getattr(configs, 'min_period', 3)
#         self.drop2d   = getattr(configs, 'conv2d_dropout', 0.0)
#
#         self.decomp   = SeriesDecomp(self.kernel)
#         # 相位鲁棒的环形对齐层（1×3），也可再堆一层 dilation=2
#         self.align1   = CircularConvAlongP(self.d_model, kernel_size=3, dilation=1)
#         self.align2   = CircularConvAlongP(self.d_model, kernel_size=3, dilation=2)
#         # 多尺度 Inception 残差块
#         self.conv2d   = Inception2dResBlock(self.d_model, drop=self.drop2d)
#
#         # 内容感知门控（样本级标量）
#         hidden = max(8, self.d_model // 4)
#         self.gate_mlp = nn.Sequential(
#             nn.Linear(self.d_model, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, 1)
#         )
#
#         self.eps = 1e-8
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, T, C=d_model]
#         """
#         B, T, C = x.shape
#         assert C == self.d_model, "Input last dim must equal d_model"
#
#         # 分解
#         s, t = self.decomp(x)  # [B,T,C]
#
#         # 候选周期与先验
#         idx, w_prior = auto_correlation(s, self.k, min_p=self.min_p, max_p=T // 2)  # [B,k], [B,k]
#
#         # 聚合容器（张量化 index_add_）
#         agg_num = torch.zeros_like(x)                  # [B,T,C]
#         agg_den_1d = x.new_zeros(B)                    # [B]
#
#         # 以 unique(p) 为桶，批量处理所有 (b,j)
#         unique_p = torch.unique(idx)
#         for pv in unique_p.tolist():
#             # 选择该周期的所有 (b,j)
#             mask = (idx == pv)                         # [B,k]
#             if not mask.any():
#                 continue
#             b_idx, j_idx = mask.nonzero(as_tuple=True)   # [m], [m]
#             m = b_idx.numel()
#
#             # 选出对应的季节项与先验
#             sb = s[b_idx]                               # [m,T,C]
#             wb = w_prior[b_idx, j_idx].view(m, 1, 1)    # [m,1,1]
#
#             # 折叠为 2D
#             z, _T = _fold_2d_reflect(sb, int(pv))       # [m,C,cyc,p]
#
#             # 相位鲁棒：两层环形卷积（等价软对齐）
#             z = self.align1(z)
#             z = self.align2(z)
#
#             # 多尺度 2D 残差块
#             z = self.conv2d(z)                          # [m,C,cyc,p]
#
#             # 展回 1D
#             y = _unfold_1d(z, _T)                       # [m,T,C]
#
#             # 内容门控（样本级标量）
#             gate = torch.sigmoid(self.gate_mlp(y.mean(dim=1)))  # [m,1]
#             score = torch.log(wb + self.eps) + torch.log(gate.view(m, 1, 1) + self.eps)  # [m,1,1]
#             score_exp = torch.exp(score)                 # [m,1,1]
#
#             # 累加：num 和 den
#             contrib = y * score_exp                      # [m,T,C]
#             agg_num.index_add_(0, b_idx, contrib)        # 按样本聚合
#             agg_den_1d.index_add_(0, b_idx, score_exp.view(-1))  # [B]
#
#         # 归一化融合
#         agg = agg_num / (agg_den_1d.view(B, 1, 1) + self.eps)
#
#         # 加回趋势，并保留季节残差，不重复趋势
#         out = agg + t
#         return out + (x - t)
#
#
# # ------------------------------------------------------------
# # 7) 分类模型：投影 + N×TimesBlock + 注意力池化 + 分类头
# # ------------------------------------------------------------
# class Model(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.d_model   = configs.d_model
#         self.num_class = configs.num_class
#         self.dropout_p = configs.dropout
#
#         # 输入特征 -> d_model
#         self.project = nn.Linear(configs.enc_in, self.d_model)
#
#         # 堆叠 TimesBlock
#         self.blocks = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
#
#         # Post-Norm（如训练不稳可切换 Pre-Norm）
#         self.norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(configs.e_layers)])
#
#         # 注意力池化（可学习 query）
#         self.pool_q = nn.Parameter(torch.randn(1, 1, self.d_model))
#
#         # 分类头
#         self.dropout    = nn.Dropout(self.dropout_p)
#         self.classifier = nn.Linear(self.d_model, self.num_class)
#
#     def attention_pool(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, T, d]
#         return: [B, d]
#         """
#         B, T, d = x.shape
#         q = self.pool_q.expand(B, -1, -1)                       # [B,1,d]
#         att = torch.matmul(q, x.transpose(1, 2)) / (d ** 0.5)   # [B,1,T]
#         att = torch.softmax(att, dim=-1)
#         x_pooled = torch.matmul(att, x).squeeze(1)              # [B,d]
#         return x_pooled
#
#     def forward(self, x_enc: torch.Tensor, *args) -> torch.Tensor:
#         """
#         x_enc: [B, T, enc_in]
#         """
#         # 投影
#         x = self.project(x_enc)                                 # [B,T,d]
#
#         # N 层 TimesBlock（每层内自带残差）
#         for blk, ln in zip(self.blocks, self.norms):
#             x = ln(blk(x))                                      # [B,T,d]
#
#         # 注意力池化
#         x = self.attention_pool(x)                              # [B,d]
#         x = self.dropout(x)
#
#         # 分类输出
#         out = self.classifier(x)                                # [B,num_class]
#         return out

# ------------------------------------------------------------
# version2
# ------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
#
# # ------------------------------------------------------------
# # 1) 序列分解：SeriesDecomp（中心移动平均）
# # ------------------------------------------------------------
# class SeriesDecomp(nn.Module):
#     def __init__(self, kernel_size: int):
#         super().__init__()
#         self.avg_pool = nn.AvgPool1d(
#             kernel_size=kernel_size, stride=1,
#             padding=(kernel_size - 1) // 2, count_include_pad=False
#         )
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         x: [B, T, C]
#         return: seasonal, trend
#         """
#         trend = self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B,T,C]
#         seasonal = x - trend
#         return seasonal, trend
#
#
# # ------------------------------------------------------------
# # 2) 自相关（逐样本）：Auto-Correlation
# #    限制候选周期范围，避免极小/极大周期退化
# # ------------------------------------------------------------
# def auto_correlation(x: torch.Tensor, k: int, min_p: int = 3, max_p: int = None):
#     """
#     x: [B, T, C]
#     return:
#       idx: [B, k]   每个样本的周期(滞后)
#       w  : [B, k]   对应 softmax 权重（来自自相关峰值，仅作先验，不直接做最终融合）
#     """
#     B, T, C = x.shape
#     if max_p is None:
#         max_p = max(1, T // 2)
#
#     # 频域功率谱（逐样本）
#     Xf = torch.fft.rfft(x, dim=1)              # [B, F, C]
#     P = (Xf * torch.conj(Xf)).real             # [B, F, C]
#     Pm = P.mean(dim=-1)                        # [B, F]
#     r = torch.fft.irfft(Pm, n=T, dim=1)        # [B, T]
#     # 屏蔽非法滞后：0、[1, min_p-1]、(max_p, T-1]
#     r[:, 0] = float('-inf')
#     if min_p > 1:
#         r[:, 1:min_p] = float('-inf')
#     if max_p + 1 < T:
#         r[:, max_p + 1:] = float('-inf')
#
#     vals, idx = torch.topk(r, k, dim=1)        # [B, k]
#     w = F.softmax(vals, dim=1)                 # [B, k] 先验权重
#     return idx, w
#
#
# # ------------------------------------------------------------
# # 3) 轻量 2D 卷积块：Conv-BN-GELU-Conv-BN + 残差
# # ------------------------------------------------------------
# class Conv2dResBlock(nn.Module):
#     def __init__(self, channels: int, drop: float = 0.0):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.GELU(),
#             nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#         )
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         return z + self.block(z)               # 残差
#
#
# # ------------------------------------------------------------
# # 4) TimesBlock（逐样本周期 + 只用 T + 反射填充 + 2D 残差块）
# #    修复点：
# #      - 取消时间域 roll（相位对齐由2D卷积等变性/融合解决）
# #      - 剪枝极端周期（auto_correlation 内已处理）
# #      - 融合加入内容感知门控（gate）而非仅依赖自相关先验
# #      - 残差不重复叠加趋势：return out + (x - t)
# # ------------------------------------------------------------
# class TimesBlock(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.k        = configs.top_k
#         self.kernel   = configs.moving_avg
#         self.d_model  = configs.d_model
#         self.min_p    = getattr(configs, 'min_period', 3)
#         self.drop2d   = getattr(configs, 'conv2d_dropout', 0.0)
#
#         self.decomp   = SeriesDecomp(self.kernel)
#         self.conv2d   = Conv2dResBlock(self.d_model, drop=self.drop2d)
#
#         # 内容感知门控 MLP：来自分支输出的全局特征 -> 标量 gate
#         hidden = max(8, self.d_model // 4)
#         self.gate_mlp = nn.Sequential(
#             nn.Linear(self.d_model, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, 1)   # 输出每个样本的标量 gate
#         )
#
#         self.eps = 1e-8
#
#     @staticmethod
#     def _fold_2d(x_1d: torch.Tensor, p: int) -> Tuple[torch.Tensor, int]:
#         """
#         x_1d: [b, T, C]
#         p:    period
#         return z: [b, C, cyc, p], T_orig
#         """
#         b, T, C = x_1d.shape
#         pad = ((T + p - 1) // p) * p - T
#         if pad > 0:
#             # 反射填充在时间维（dim=1）
#             x_1d = F.pad(x_1d, (0, 0, 0, pad), mode='reflect')
#         cyc = (T + pad) // p
#         z = x_1d.reshape(b, cyc, p, C).permute(0, 3, 1, 2)  # [b,C,cyc,p]
#         return z, T
#
#     @staticmethod
#     def _unfold_1d(z_2d: torch.Tensor, T: int) -> torch.Tensor:
#         """
#         z_2d: [b, C, cyc, p]
#         return y: [b, T, C]
#         """
#         b, C, cyc, p = z_2d.shape
#         y = z_2d.permute(0, 2, 3, 1).reshape(b, cyc * p, C)[:, :T, :]  # [b,T,C]
#         return y
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, T, C=d_model]
#         """
#         B, T, C = x.shape
#         assert C == self.d_model, "Input last dim must equal d_model"
#
#         # 分解
#         s, t = self.decomp(x)  # [B,T,C]
#
#         # 逐样本周期与自相关先验权重（范围裁剪见函数内部）
#         idx, w_prior = auto_correlation(s, self.k, min_p=self.min_p, max_p=T // 2)  # [B,k], [B,k]
#
#         # 准备融合的分子/分母（做归一化的内容感知融合）
#         agg_num = torch.zeros_like(x)                    # ∑ y * exp(score)
#         agg_den = torch.zeros(B, 1, 1, device=x.device, dtype=x.dtype)  # ∑ exp(score)
#
#         # 按第 j 个候选周期处理
#         for j in range(self.k):
#             p_b = idx[:, j]  # [B]
#             unique_p = p_b.unique()
#             for pv in unique_p.tolist():
#                 sel = (p_b == pv)
#                 if not torch.any(sel):
#                     continue
#
#                 sb = s[sel]                               # [b,T,C]
#                 wb = w_prior[sel, j].view(-1, 1, 1)       # [b,1,1] 先验（不做最终权重，参与打分）
#
#                 # 直接折叠为 2D（不再做时间域 roll）
#                 z, _T = self._fold_2d(sb, int(pv))        # [b,C,cyc,p]
#
#                 # 2D 残差卷积
#                 z = self.conv2d(z)                        # [b,C,cyc,p]
#
#                 # 展回 1D
#                 y = self._unfold_1d(z, _T)                # [b,T,C]
#
#                 # 内容感知门控：对分支输出做全局均值 -> MLP -> 标量
#                 # 用展回后的时域特征，聚合时间得到 [b,C]，再 MLP -> [b,1]
#                 gate = self.gate_mlp(y.mean(dim=1))       # [b,1]
#                 gate = torch.sigmoid(gate).view(-1, 1, 1) # [b,1,1]
#
#                 # 基于先验 + 内容门控的打分：score = log(w_prior) + log(gate)
#                 # 训练更稳：在 log 里加 eps
#                 score = torch.log(wb + self.eps) + torch.log(gate + self.eps)  # [b,1,1]
#                 score_exp = torch.exp(score)                                    # [b,1,1]
#
#                 # 累加分子/分母
#                 agg_num[sel] = agg_num[sel] + y * score_exp
#                 agg_den[sel] = agg_den[sel] + score_exp.squeeze(-1).squeeze(-1).unsqueeze(-1).unsqueeze(-1)
#
#         # 归一化融合
#         agg = agg_num / (agg_den + self.eps)
#
#         # 加回趋势
#         out = agg + t
#
#         # 关键修复：保持残差但不重复将趋势加入两次
#         # x = seasonal + trend，所以 (x - t) = seasonal
#         return out + (x - t)
#
#
# # ------------------------------------------------------------
# # 5) 分类模型：投影 + N×TimesBlock + 注意力池化 + 分类头
# # ------------------------------------------------------------
# class Model(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.d_model   = configs.d_model
#         self.num_class = configs.num_class
#         self.dropout_p = configs.dropout
#
#         # 输入特征 -> d_model
#         self.project = nn.Linear(configs.enc_in, self.d_model)
#
#         # 堆叠 TimesBlock
#         self.blocks = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
#
#         # Post-Norm（保持你的设置；若训练不稳可改为 Pre-Norm）
#         self.norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(configs.e_layers)])
#
#         # 注意力池化
#         self.pool_q = nn.Parameter(torch.randn(1, 1, self.d_model))
#
#         # 分类头
#         self.dropout    = nn.Dropout(self.dropout_p)
#         self.classifier = nn.Linear(self.d_model, self.num_class)
#
#     def attention_pool(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, T, d]
#         return: [B, d]
#         """
#         B, T, d = x.shape
#         q = self.pool_q.expand(B, -1, -1)                       # [B,1,d]
#         att = torch.matmul(q, x.transpose(1, 2)) / (d ** 0.5)   # [B,1,T]
#         att = torch.softmax(att, dim=-1)
#         x_pooled = torch.matmul(att, x).squeeze(1)              # [B,d]
#         return x_pooled
#
#     def forward(self, x_enc: torch.Tensor, *args) -> torch.Tensor:
#         """
#         x_enc: [B, T, enc_in]
#         """
#         # 投影
#         x = self.project(x_enc)                                 # [B,T,d]
#
#         # N 层 TimesBlock（每层内自带残差）
#         for blk, ln in zip(self.blocks, self.norms):
#             x = ln(blk(x))                                      # [B,T,d]
#
#         # 注意力池化
#         x = self.attention_pool(x)                              # [B,d]
#         x = self.dropout(x)
#
#         # 分类输出
#         out = self.classifier(x)                                # [B,num_class]
#         return out

# ------------------------------------------------------------
# version1
# ------------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
#
# # ------------------------------------------------------------
# # 1) 序列分解：SeriesDecomp（中心移动平均）
# # ------------------------------------------------------------
# class SeriesDecomp(nn.Module):
#     def __init__(self, kernel_size: int):
#         super().__init__()
#         self.avg_pool = nn.AvgPool1d(
#             kernel_size=kernel_size, stride=1,
#             padding=(kernel_size - 1) // 2, count_include_pad=False
#         )
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         x: [B, T, C]
#         return: seasonal, trend  (same shape as x)
#         """
#         trend = self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B,T,C]
#         seasonal = x - trend
#         return seasonal, trend
#
#
# # ------------------------------------------------------------
# # 2) 自相关（逐样本）：Auto-Correlation
# #    只用当前长度 T；返回每个样本的 top-k 周期与权重
# # ------------------------------------------------------------
# def auto_correlation(x: torch.Tensor, k: int):
#     """
#     x: [B, T, C]
#     return:
#       idx: [B, k]   每个样本的周期(滞后)
#       w  : [B, k]   对应 softmax 权重
#     """
#     B, T, C = x.shape
#     Xf = torch.fft.rfft(x, dim=1)             # [B, F, C]
#     P = (Xf * torch.conj(Xf)).real            # [B, F, C]
#     Pm = P.mean(dim=-1)                       # [B, F]  # 逐样本功率谱
#     r = torch.fft.irfft(Pm, n=T, dim=1)       # [B, T]  # 逐样本自相关
#     r[:, 0] = float('-inf')                   # 禁止零滞后
#     vals, idx = torch.topk(r, k, dim=1)       # [B, k]
#     w = F.softmax(vals, dim=1)                # [B, k]
#     return idx, w
#
#
# # ------------------------------------------------------------
# # 3) 轻量 2D 卷积块：Conv-BN-GELU-Conv-BN + 残差
# # ------------------------------------------------------------
# class Conv2dResBlock(nn.Module):
#     def __init__(self, channels: int):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.GELU(),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#         )
#
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         return z + self.block(z)               # 残差
#
#
# # ------------------------------------------------------------
# # 4) TimesBlock（逐样本周期 + 只用 T + 反射填充 + 2D 残差块）
# # ------------------------------------------------------------
# class TimesBlock(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.k       = configs.top_k
#         self.kernel  = configs.moving_avg
#         self.d_model = configs.d_model
#
#         self.decomp  = SeriesDecomp(self.kernel)
#         self.conv2d  = Conv2dResBlock(self.d_model)
#
#     @staticmethod
#     def _fold_2d(x_1d: torch.Tensor, p: int) -> torch.Tensor:
#         """
#         x_1d: [b, T, C]
#         p:    period
#         return z: [b, C, cyc, p]
#         """
#         b, T, C = x_1d.shape
#         pad = ((T + p - 1) // p) * p - T
#         if pad > 0:
#             # 反射填充在时间维（dim=1）
#             x_1d = F.pad(x_1d, (0, 0, 0, pad), mode='reflect')  # (C_pad=0, T_pad=pad)
#         cyc = (T + pad) // p
#         z = x_1d.reshape(b, cyc, p, C).permute(0, 3, 1, 2)      # [b,C,cyc,p]
#         return z, T  # 返回原始 T 便于后续裁剪
#
#     @staticmethod
#     def _unfold_1d(z_2d: torch.Tensor, T: int) -> torch.Tensor:
#         """
#         z_2d: [b, C, cyc, p]
#         return y: [b, T, C]
#         """
#         b, C, cyc, p = z_2d.shape
#         y = z_2d.permute(0, 2, 3, 1).reshape(b, cyc * p, C)[:, :T, :]  # [b,T,C]
#         return y
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, T, C=d_model]
#         """
#         B, T, C = x.shape
#         assert C == self.d_model, "Input last dim must equal d_model"
#
#         # 分解
#         s, t = self.decomp(x)  # [B,T,C]
#         # 逐样本周期与权重
#         idx, w = auto_correlation(s, self.k)  # [B,k], [B,k]
#
#         # 聚合容器（始终使用当前 T）
#         agg = torch.zeros_like(x)
#
#         # 按第 j 个候选周期累加
#         for j in range(self.k):
#             p_b = idx[:, j]            # [B]
#             # 将相同 p 的样本分组以做并行
#             unique_p = p_b.unique()
#             for pv in unique_p.tolist():
#                 sel = (p_b == pv)
#                 if not torch.any(sel):
#                     continue
#                 sb = s[sel]                                # [b,T,C]
#                 wb = w[sel, j].view(-1, 1, 1)              # [b,1,1]
#                 # 对齐（向前滚动 pv）
#                 rolled = torch.roll(sb, shifts=-int(pv), dims=1)
#                 # 折叠 -> 2D
#                 z, _T = self._fold_2d(rolled, int(pv))     # [b,C,cyc,p]
#                 # 2D 残差卷积
#                 z = self.conv2d(z)                         # [b,C,cyc,p]
#                 # 展回 1D 并裁剪到 T
#                 y = self._unfold_1d(z, _T)                 # [b,T,C]
#                 # 权重加权并累加
#                 agg[sel] = agg[sel] + y * wb
#
#         # 加回趋势 + 残差
#         out = agg + t
#         return out + x
#
#
# # ------------------------------------------------------------
# # 5) 分类模型：投影 + N×TimesBlock + 注意力池化 + 分类头
# #    （用注意力池化替换原来的 GAP）
# # ------------------------------------------------------------
# class Model(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.d_model   = configs.d_model
#         self.num_class = configs.num_class
#         self.dropout_p = configs.dropout
#
#         # 输入特征 -> d_model
#         self.project = nn.Linear(configs.enc_in, self.d_model)
#
#         # 堆叠 TimesBlock
#         self.blocks = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
#
#         # Post-Norm（保持你原先用法；如果不稳可切 Pre-Norm）
#         self.norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(configs.e_layers)])
#
#         # 注意力池化
#         self.pool_q = nn.Parameter(torch.randn(1, 1, self.d_model))
#
#         # 分类头
#         self.dropout    = nn.Dropout(self.dropout_p)
#         self.classifier = nn.Linear(self.d_model, self.num_class)
#
#     def attention_pool(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, T, d]
#         return: [B, d]
#         """
#         B, T, d = x.shape
#         q = self.pool_q.expand(B, -1, -1)                  # [B,1,d]
#         att = torch.matmul(q, x.transpose(1, 2)) / (d ** 0.5)  # [B,1,T]
#         att = torch.softmax(att, dim=-1)
#         x_pooled = torch.matmul(att, x).squeeze(1)         # [B,d]
#         return x_pooled
#
#     def forward(self, x_enc: torch.Tensor, *args) -> torch.Tensor:
#         """
#         x_enc: [B, T, enc_in]
#         """
#         # 投影
#         x = self.project(x_enc)                            # [B,T,d]
#         # N 层 TimesBlock
#         for blk, ln in zip(self.blocks, self.norms):
#             x = ln(blk(x))                                 # [B,T,d]
#         # 注意力池化
#         x = self.attention_pool(x)                         # [B,d]
#         x = self.dropout(x)
#         # 分类
#         out = self.classifier(x)                           # [B,num_class]
#         return out


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
