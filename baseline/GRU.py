# models/GRU.py
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    GRU 分类模型（兼容 Exp_Classification）
      forward(x, padding_mask=None, x_mark=None, y_mark=None) -> logits [B, num_class]

    超参（args）：
      - enc_in:      输入维度（必填）
      - num_class:   类别数（必填）
      - gru_hidden:  GRU 隐藏维度（默认 128）
      - gru_layers:  GRU 堆叠层数（默认 2）
      - gru_dropout: Dropout 概率（默认 0.2；层数>1时才生效）
      - gru_pool:    池化方式 'mean' | 'last' | 'max'（默认 'mean'）
      - bidirectional: 是否双向（默认 False）
    """
    def __init__(self, args):
        super().__init__()
        self.enc_in    = getattr(args, 'enc_in', None)
        self.num_class = getattr(args, 'num_class', None)
        assert self.enc_in is not None and self.num_class is not None, \
            "args.enc_in 和 args.num_class 必须提供"

        hidden_dim    = getattr(args, 'gru_hidden', 128)
        num_layers    = getattr(args, 'gru_layers', 2)
        dropout_prob  = getattr(args, 'gru_dropout', 0.2)
        self.pooling  = getattr(args, 'gru_pool', 'mean').lower()
        self.bidirect = getattr(args, 'bidirectional', False)

        assert self.pooling in ('mean', 'last', 'max')

        self.gru = nn.GRU(
            input_size=self.enc_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
            bidirectional=self.bidirect
        )

        out_dim = hidden_dim * (2 if self.bidirect else 1)
        mid_dim = max(64, out_dim // 2)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(mid_dim, self.num_class)
        )

    # ---------- 池化工具函数 ----------
    @staticmethod
    def _masked_mean(x_blc, mask_bl=None, eps: float = 1e-8):
        if mask_bl is None:
            return x_blc.mean(dim=1)
        w = mask_bl.float().clamp(0, 1)  # [B, L]
        denom = w.sum(dim=1, keepdim=True).clamp_min(eps)
        return (x_blc * w.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _masked_max(x_blc, mask_bl=None):
        if mask_bl is None:
            return x_blc.max(dim=1).values
        very_neg = torch.finfo(x_blc.dtype).min
        m = mask_bl.unsqueeze(-1).bool()
        x_masked = x_blc.masked_fill(~m, very_neg)
        return x_masked.max(dim=1).values

    @staticmethod
    def _gather_last(x_blc, mask_bl=None):
        B, L, C = x_blc.shape
        if mask_bl is None:
            idx = torch.full((B,), L - 1, device=x_blc.device, dtype=torch.long)
        else:
            lengths = mask_bl.long().sum(dim=1)
            idx = (lengths - 1).clamp(min=0)
        arange_b = torch.arange(B, device=x_blc.device)
        return x_blc[arange_b, idx, :]

    # ---------- 前向 ----------
    def forward(self, x, padding_mask=None, x_mark=None, y_mark=None):
        """
        x: [B, L, C]
        padding_mask: [B, L] (1/True=有效, 0/False=padding)，可选
        """
        gru_out, _ = self.gru(x)  # [B, L, H] (或 [B, L, 2H] 若 bidirectional)

        if self.pooling == 'mean':
            feat = self._masked_mean(gru_out, padding_mask)
        elif self.pooling == 'max':
            feat = self._masked_max(gru_out, padding_mask)
        else:  # last
            feat = self._gather_last(gru_out, padding_mask)

        logits = self.classifier(feat)  # [B, num_class]
        return logits
