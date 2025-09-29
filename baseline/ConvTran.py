# models/ConvTran.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    ConvTran: Convolution + Transformer for Time Series Classification
    兼容 Exp_Classification: outputs = model(batch_x, padding_mask, None, None)

    必要参数（args）：
      - enc_in: 输入特征维度
      - num_class: 类别数

    可选参数：
      - conv_channels: Conv 输出通道数（默认 64）
      - conv_kernel: Conv 卷积核大小（默认 5，奇数推荐）
      - conv_stride: Conv 步幅（默认 1）
      - d_model: Transformer 维度（默认 128）
      - nhead: Transformer 多头数（默认 8）
      - e_layers: Transformer encoder 层数（默认 2）
      - d_ff: FFN 隐层大小（默认 256）
      - dropout: dropout 概率（默认 0.1）
      - pool: 池化方式 'mean' | 'cls' | 'max'（默认 mean）
    """

    def __init__(self, args):
        super().__init__()
        self.enc_in    = getattr(args, "enc_in", None)
        self.num_class = getattr(args, "num_class", None)
        assert self.enc_in and self.num_class, "args.enc_in 和 args.num_class 必须提供"

        conv_channels = getattr(args, "conv_channels", 64)
        conv_kernel   = getattr(args, "conv_kernel", 5)
        conv_stride   = getattr(args, "conv_stride", 1)
        d_model       = getattr(args, "d_model", 128)
        nhead         = getattr(args, "nhead", 8)
        e_layers      = getattr(args, "e_layers", 2)
        d_ff          = getattr(args, "d_ff", 256)
        dropout       = getattr(args, "dropout", 0.1)
        self.pool     = getattr(args, "pool", "mean")

        # --------- 卷积特征提取 ---------
        pad = conv_kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(self.enc_in, conv_channels,
                      kernel_size=conv_kernel,
                      stride=conv_stride,
                      padding=pad,
                      bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True)
        )

        # 投影到 Transformer d_model
        self.proj = nn.Linear(conv_channels, d_model)

        # --------- Transformer Encoder ---------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # [CLS] token（仅在 pool=cls 时使用）
        if self.pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        else:
            self.cls_token = None

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, max(64, d_model // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(64, d_model // 2), self.num_class)
        )

    def forward(self, x, padding_mask=None, x_mark=None, y_mark=None):
        """
        x: [B, L, C]
        padding_mask: [B, L] (1=有效, 0=padding)
        """
        B, L, C = x.shape

        # --------- 卷积 ---------
        x = x.transpose(1, 2)              # [B, C, L]
        x = self.conv(x)                   # [B, conv_channels, L]
        x = x.transpose(1, 2)              # [B, L, conv_channels]
        x = self.proj(x)                   # [B, L, d_model]

        # --------- 可选 CLS token ---------
        if self.pool == "cls":
            cls_tok = self.cls_token.expand(B, -1, -1)   # [B, 1, d_model]
            x = torch.cat([cls_tok, x], dim=1)           # [B, L+1, d_model]
            if padding_mask is not None:
                pad_col = torch.ones(B, 1, device=x.device, dtype=padding_mask.dtype)
                padding_mask = torch.cat([pad_col, padding_mask], dim=1)  # [B, L+1]

        # --------- Transformer Encoder ---------
        if padding_mask is not None:
            key_padding_mask = ~padding_mask.bool()  # Transformer: True=要mask
        else:
            key_padding_mask = None
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, L(+1), d_model]

        # --------- 池化 ---------
        if self.pool == "mean":
            if padding_mask is None:
                feat = out.mean(dim=1)                   # [B, d_model]
            else:
                w = padding_mask.float().clamp(0, 1)
                feat = (out * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        elif self.pool == "max":
            if padding_mask is None:
                feat = out.max(dim=1).values
            else:
                very_neg = torch.finfo(out.dtype).min
                m = padding_mask.unsqueeze(-1).bool()
                feat = out.masked_fill(~m, very_neg).max(dim=1).values
        else:  # cls
            feat = out[:, 0, :]                          # [B, d_model]

        # --------- 分类 ---------
        logits = self.classifier(feat)                   # [B, num_class]
        return logits
