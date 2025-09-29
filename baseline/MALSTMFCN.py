# MALSTM_FCN.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------- 工具：掩码池化 -----------------------
def masked_mean_pool(x_blc, mask_bl=None, eps: float = 1e-8):
    """
    x_blc: [B, L, C]
    mask_bl: [B, L] (1/True=有效, 0/False=padding) 或 None
    返回: [B, C]
    """
    if mask_bl is None:
        return x_blc.mean(dim=1)
    w = mask_bl.float().clamp(0, 1)                         # [B, L]
    denom = w.sum(dim=1, keepdim=True).clamp_min(eps)       # [B, 1]
    return (x_blc * w.unsqueeze(-1)).sum(dim=1) / denom     # [B, C]


def masked_gmp_pool(x_blc, mask_bl=None):
    """
    全局最大池化，mask 位置置为极小再 max
    """
    if mask_bl is None:
        return x_blc.max(dim=1).values
    very_neg = torch.finfo(x_blc.dtype).min
    m = mask_bl.unsqueeze(-1).bool()
    return x_blc.masked_fill(~m, very_neg).max(dim=1).values


# ----------------------- 多头注意力池化 -----------------------
class MultiHeadTemporalAttention(nn.Module):
    """
    时间维多头注意力池化（每个头一个可学习查询向量）
    输入: x [B, L, H], mask [B, L]
    输出: ctx [B, heads*H], attn [B, heads, L]
    """
    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.proj = nn.Linear(hidden, hidden * heads, bias=True)  # 共享映射，后分头
        self.v = nn.Parameter(torch.randn(heads, hidden))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.v)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_blh, mask_bl=None):
        B, L, H = x_blh.shape
        u = torch.tanh(self.proj(x_blh))                      # [B, L, H*heads]
        u = u.view(B, L, self.heads, H).permute(0, 2, 1, 3)   # [B, heads, L, H]
        scores = torch.einsum('bhlt,ht->bhl', u, self.v)      # [B, heads, L]

        if mask_bl is not None:
            scores = scores.masked_fill(~mask_bl[:, None, :].bool(), float('-inf'))

        attn = torch.softmax(scores, dim=-1)                  # [B, heads, L]
        attn = self.drop(attn)
        ctx = torch.einsum('bhl,bld->bhd', attn, x_blh)       # [B, heads, H]
        ctx = ctx.reshape(B, self.heads * H)                  # [B, heads*H]
        return ctx, attn


# ----------------------- SE 模块（可选） -----------------------
class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),     # [B, C, 1]
            nn.Flatten(),                # [B, C]
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x_bcl):
        w = self.fc(x_bcl)               # [B, C]
        return x_bcl * w.unsqueeze(-1)   # [B, C, L]


# ----------------------- FCN 分支 -----------------------
class FCNBackbone(nn.Module):
    """
    经典 TSC 的 FCN 堆叠: Conv-BN-ReLU x 3 (+ 可选 SE) + GAP
    期望输入 [B, L, C]，内部转 [B, C, L]
    """
    def __init__(self, in_ch: int, channels=(128, 256, 128), kernels=(7, 5, 3),
                 use_se: bool = True, dropout: float = 0.1):
        super().__init__()
        C1, C2, C3 = channels
        k1, k2, k3 = kernels

        layers = []
        last = in_ch
        for i, (outc, k) in enumerate(zip((C1, C2, C3), (k1, k2, k3))):
            pad = k // 2  # 奇数核时等同保长，偶数核时长度会+/-1，这里后续做对齐保护
            layers += [
                nn.Conv1d(last, outc, kernel_size=k, padding=pad, bias=False),
                nn.BatchNorm1d(outc),
                nn.ReLU(inplace=True),
            ]
            if i == 1 and use_se:
                layers += [SqueezeExcite1d(outc)]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = outc
        self.conv = nn.Sequential(*layers)
        self.out_ch = last

    def forward(self, x_blc, mask_bl=None):
        # [B, L, C] -> [B, C, L]
        x = x_blc.transpose(1, 2).contiguous()
        x = self.conv(x)                          # [B, C3, L']
        x_blc2 = x.transpose(1, 2).contiguous()   # [B, L', C3]

        # ------- 安全对齐：若 L' 与 mask 的 L 不一致，按较短值截断 -------
        if mask_bl is not None:
            Lp = x_blc2.size(1)
            Lm = mask_bl.size(1)
            if Lp != Lm:
                Lmin = min(Lp, Lm)
                x_blc2 = x_blc2[:, :Lmin, :]
                mask_bl = mask_bl[:, :Lmin]
        # ----------------------------------------------------------------

        feat = masked_mean_pool(x_blc2, mask_bl)  # [B, C3]
        return feat


# ----------------------- 顶层模型 -----------------------
class Model(nn.Module):
    """
    MALSTM-FCN（Multi-head Attention LSTM + FCN）
    forward(x, padding_mask=None, x_mark=None, y_mark=None) -> logits [B, num_class]

    必要参数（args）：
      - enc_in:         输入特征维度
      - num_class:      类别数

    可选超参（默认值）：
      - lstm_hidden=128, lstm_layers=2, lstm_dropout=0.2, bidirectional=True
      - attn_heads=4, attn_dropout=0.1
      - fcn_channels=(128,256,128), fcn_kernels=(7,5,3), fcn_dropout=0.1, fcn_use_se=True
      - cls_dropout=0.2
    """
    def __init__(self, args):
        super().__init__()
        self.enc_in    = getattr(args, 'enc_in', None)
        self.num_class = getattr(args, 'num_class', None)
        assert self.enc_in is not None and self.num_class is not None, \
            "args.enc_in 与 args.num_class 必须提供"

        # LSTM 分支
        lstm_hidden   = getattr(args, 'lstm_hidden', 128)
        lstm_layers   = getattr(args, 'lstm_layers', 2)
        lstm_dropout  = getattr(args, 'lstm_dropout', 0.2)
        bidirect      = getattr(args, 'bidirectional', True)

        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirect
        )
        lstm_out_dim = lstm_hidden * (2 if bidirect else 1)

        # 时间维多头注意力池化
        attn_heads   = getattr(args, 'attn_heads', 4)
        attn_dropout = getattr(args, 'attn_dropout', 0.1)
        self.temporal_attn = MultiHeadTemporalAttention(
            hidden=lstm_out_dim, heads=attn_heads, dropout=attn_dropout
        )
        lstm_feat_dim = lstm_out_dim * attn_heads

        # FCN 分支
        fcn_channels = getattr(args, 'fcn_channels', (128, 256, 128))
        fcn_kernels  = getattr(args, 'fcn_kernels',  (7, 5, 3))   # 默认奇数核
        fcn_dropout  = getattr(args, 'fcn_dropout',  0.1)
        fcn_use_se   = getattr(args, 'fcn_use_se',   True)
        self.fcn = FCNBackbone(
            in_ch=self.enc_in, channels=fcn_channels, kernels=fcn_kernels,
            use_se=fcn_use_se, dropout=fcn_dropout
        )
        fcn_feat_dim = self.fcn.out_ch

        # 分类头
        cls_dropout  = getattr(args, 'cls_dropout', 0.2)
        fused_dim = lstm_feat_dim + fcn_feat_dim
        mid_dim = max(64, fused_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(mid_dim, self.num_class)
        )

    def forward(self, x, padding_mask=None, x_mark=None, y_mark=None):
        """
        x: [B, L, C]
        padding_mask: [B, L] (1/True=有效, 0/False=padding)，可为 None
        """
        # LSTM 分支
        lstm_out, _ = self.lstm(x)                                    # [B, L, H/2H]
        lstm_feat, _attn = self.temporal_attn(lstm_out, padding_mask) # [B, heads*H]

        # FCN 分支
        fcn_feat = self.fcn(x, padding_mask)                          # [B, C3]

        # 融合 & 分类
        fused = torch.cat([lstm_feat, fcn_feat], dim=-1)              # [B, heads*H + C3]
        logits = self.classifier(fused)                                # [B, num_class]
        return logits
