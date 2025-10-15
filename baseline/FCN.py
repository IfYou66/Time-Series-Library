# models/FCN.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ 掩码友好池化 ------------------
def masked_mean_pool(x_blc, mask_bl=None, eps: float = 1e-8):
    """
    按照掩码进行平均池化，忽略 padding 部分
    x_blc: [B, L, C] (batch_size, sequence_length, feature_dim)
    mask_bl: [B, L] (padding_mask)，0 = padding, 1 = valid data (optional)
    """
    if mask_bl is None:
        return x_blc.mean(dim=1)
    w = mask_bl.float().clamp(0, 1)  # [B, L] 转成 [B, L] 类型的掩码
    denom = w.sum(dim=1, keepdim=True).clamp_min(eps)  # [B, 1] sum(mask) + eps 避免除 0
    return (x_blc * w.unsqueeze(-1)).sum(dim=1) / denom  # [B, C]


# ------------------ Squeeze-and-Excitation (SE) ------------------
class SqueezeExcite1d(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D Data
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, C, 1]
            nn.Flatten(),  # [B, C]
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x_bcl):
        w = self.fc(x_bcl)  # [B, C]
        return x_bcl * w.unsqueeze(-1)  # [B, C, L]


# ------------------ FCN 分支 ------------------
class FCNBackbone(nn.Module):
    """
    FCN: Fully Convolutional Network for Time Series Classification
    """

    def __init__(self, in_ch: int, channels=(128, 256, 128), kernels=(8, 5, 3),
                 dropout: float = 0.1, use_se: bool = True):
        super().__init__()

        layers = []
        last = in_ch
        for i, (outc, k) in enumerate(zip(channels, kernels)):
            pad = k // 2  # 使用 padding 保长
            layers += [
                nn.Conv1d(last, outc, kernel_size=k, padding=pad, bias=False),
                nn.BatchNorm1d(outc),
                nn.ReLU(inplace=True),
            ]
            if use_se and i == 1:  # 使用 Squeeze-and-Excitation 结构
                layers += [SqueezeExcite1d(outc)]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = outc
        self.conv = nn.Sequential(*layers)
        self.out_ch = last

    def forward(self, x_blc, mask_bl=None):
        # [B, L, C] -> [B, C, L]
        x = x_blc.transpose(1, 2).contiguous()
        x = self.conv(x)  # [B, C_out, L']
        x_blc2 = x.transpose(1, 2)  # [B, L', C_out]

        # ------- 安全对齐：若 L' 与 mask 的 L 不一致，按较短值截断 -------
        if mask_bl is not None:
            Lp = x_blc2.size(1)
            Lm = mask_bl.size(1)
            if Lp != Lm:
                Lmin = min(Lp, Lm)
                x_blc2 = x_blc2[:, :Lmin, :]
                mask_bl = mask_bl[:, :Lmin]
        # ----------------------------------------------------------------

        feat = masked_mean_pool(x_blc2, mask_bl)  # [B, C_out]
        return feat


class Model(nn.Module):
    """
    FCN: Fully Convolutional Network for Time Series Classification
    兼容 Exp_Classification：
      outputs = model(batch_x, padding_mask, None, None)

    必须传入的超参数：
      - enc_in:     输入特征维度
      - num_class:  类别数

    可选超参数（默认值写在括号内）：
      - fcn_channels: 每层的通道数（[128, 256, 128]）
      - fcn_kernels:  每层的卷积核大小（[8, 5, 3]）
      - fcn_dropout:  每层的 dropout 比例（0.1）
      - fcn_use_se:   是否使用 SE（True）
      - head_dropout: 分类头的 dropout（0.2）
    """

    def __init__(self, args):
        super().__init__()
        self.enc_in = getattr(args, 'enc_in', None)
        self.num_class = getattr(args, 'num_class', None)
        assert self.enc_in is not None and self.num_class is not None, \
            "args.enc_in 和 args.num_class 必须提供"

        # FCN 超参数
        fcn_channels = getattr(args, 'fcn_channels', [128, 256, 128])
        fcn_kernels = getattr(args, 'fcn_kernels', [8, 5, 3])
        fcn_dropout = getattr(args, 'fcn_dropout', 0.1)
        fcn_use_se = getattr(args, 'fcn_use_se', True)
        head_dropout = getattr(args, 'head_dropout', 0.2)

        # FCN 分支
        self.fcn = FCNBackbone(
            in_ch=self.enc_in, channels=fcn_channels, kernels=fcn_kernels,
            dropout=fcn_dropout, use_se=fcn_use_se
        )

        # 分类头
        fused_dim = self.fcn.out_ch
        mid_dim = max(64, fused_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(mid_dim, self.num_class),
        )

    def forward(self, x, padding_mask=None, x_mark=None, y_mark=None):
        """
        x: [B, L, C]
        padding_mask: [B, L] (1=有效, 0=padding)
        """
        # 1) FCN 分支
        fcn_feat = self.fcn(x, padding_mask)  # [B, C_out]

        # 2) 分类
        logits = self.classifier(fcn_feat)  # [B, num_class]
        return logits
