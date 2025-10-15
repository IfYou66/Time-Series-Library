# models/Hydra.py
import torch
import torch.nn as nn


# ---------- 掩码友好的池化 ----------
def masked_mean_pool(x_blc, mask_bl=None, eps: float = 1e-8):
    # x_blc: [B, L, C]
    if mask_bl is None:
        return x_blc.mean(dim=1)
    w = mask_bl.float().clamp(0, 1)                   # [B, L]
    denom = w.sum(dim=1, keepdim=True).clamp_min(eps) # [B, 1]
    return (x_blc * w.unsqueeze(-1)).sum(dim=1) / denom


# ---------- 可选 SE ----------
class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # [B,C,1]
            nn.Flatten(),              # [B,C]
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x_bcl):
        w = self.net(x_bcl)            # [B,C]
        return x_bcl * w.unsqueeze(-1) # [B,C,L]


# ---------- Depthwise Separable Conv 1D ----------
class DSConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, d=1, p=None):
        super().__init__()
        if p is None:
            # 使用“近似same”的 padding；奇数核完全保长，偶数核会±1，后面会与mask对齐
            p = (k // 2) * d
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=p, dilation=d,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


# ---------- 单个 Hydra 分支 ----------
class HydraBranch(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1, depth=2,
                 use_bn=True, dropout=0.0, use_se=False):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(depth):
            conv = DSConv1d(ch, out_ch, k=kernel, d=(dilation if i == 0 else 1))
            layers += [conv]
            if use_bn:
                layers += [nn.BatchNorm1d(out_ch)]
            layers += [nn.ReLU(inplace=True)]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            ch = out_ch
        if use_se:
            layers += [SqueezeExcite1d(out_ch)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_bcl):
        return self.net(x_bcl)  # [B, C_out, L']


# ---------- 顶层模型 ----------
class Model(nn.Module):
    """
    Hydra: Multi-branch 1D CNN for Time-Series Classification

    兼容 Exp_Classification：
      logits = model(batch_x, padding_mask, None, None)

    必需 args：
      - enc_in:     输入特征维度（C_in）
      - num_class:  类别数

    可选 args（默认值写在括号里）：
      - hydra_kernels:    不同分支的卷积核列表（[3,5,7,9]）
      - hydra_dilations:  不同分支的膨胀率列表（与 kernels 等长，默认全 1）
      - hydra_channels:   每个分支的输出通道数（128）
      - hydra_depth:      每个分支的堆叠层数（2）
      - hydra_use_bn:     是否使用 BN（True）
      - hydra_dropout:    分支内 dropout（0.1）
      - hydra_use_se:     分支末尾是否用 SE（True）
      - head_dropout:     分类头前的 dropout（0.2）
    """
    def __init__(self, args):
        super().__init__()
        self.enc_in    = getattr(args, 'enc_in', None)
        self.num_class = getattr(args, 'num_class', None)
        assert self.enc_in is not None and self.num_class is not None, \
            "args.enc_in 与 args.num_class 必须提供"

        kernels     = getattr(args, 'hydra_kernels',  [3, 5, 7, 9])
        dilations   = getattr(args, 'hydra_dilations', None)
        branch_out  = getattr(args, 'hydra_channels', 128)
        depth       = getattr(args, 'hydra_depth',    2)
        use_bn      = getattr(args, 'hydra_use_bn',   True)
        dropout     = getattr(args, 'hydra_dropout',  0.1)
        use_se      = getattr(args, 'hydra_use_se',   True)
        head_drop   = getattr(args, 'head_dropout',   0.2)

        if dilations is None:
            dilations = [1] * len(kernels)
        assert len(dilations) == len(kernels), "hydra_dilations 长度需与 hydra_kernels 相同"

        # 多分支
        self.branches = nn.ModuleList([
            HydraBranch(
                in_ch=self.enc_in,
                out_ch=branch_out,
                kernel=k,
                dilation=d,
                depth=depth,
                use_bn=use_bn,
                dropout=dropout,
                use_se=use_se
            )
            for k, d in zip(kernels, dilations)
        ])

        fused_dim = branch_out * len(self.branches)
        mid = max(64, fused_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, mid),
            nn.ReLU(inplace=True),
            nn.Dropout(head_drop),
            nn.Linear(mid, self.num_class),
        )

    def forward(self, x, padding_mask=None, x_mark=None, y_mark=None):
        """
        x: [B, L, C]
        padding_mask: [B, L] (1/True=有效, 0/False=padding)
        """
        # 转换为 [B,C,L] 以做卷积
        x_bcl = x.transpose(1, 2).contiguous()  # [B, C_in, L]

        feats = []
        for branch in self.branches:
            y = branch(x_bcl)                    # [B, C_out, L']
            y_blc = y.transpose(1, 2).contiguous()  # [B, L', C_out]

            # 健壮长度对齐：若 L' != L_mask，按较短截断
            if padding_mask is not None:
                Lp = y_blc.size(1)
                Lm = padding_mask.size(1)
                if Lp != Lm:
                    Lmin = min(Lp, Lm)
                    y_blc = y_blc[:, :Lmin, :]
                    mask  = padding_mask[:, :Lmin]
                else:
                    mask = padding_mask
            else:
                mask = None

            feats.append(masked_mean_pool(y_blc, mask))  # [B, C_out]

        fused = torch.cat(feats, dim=-1)   # [B, C_out * #branches]
        logits = self.classifier(fused)    # [B, num_class]
        return logits
