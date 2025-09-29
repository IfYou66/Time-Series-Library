# models/BiLSTM.py
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    BiLSTM 分类模型（兼容 Exp_Classification）
      forward(x, padding_mask=None, x_mark=None, y_mark=None) -> logits [B, num_class]

    需要的超参（args）：
      - enc_in:      输入维度（必填）
      - num_class:   类别数（必填）
      - lstm_hidden: LSTM 隐藏维度（默认 128；双向后通道=2*lstm_hidden）
      - lstm_layers: LSTM 层数（默认 2）
      - lstm_dropout:Dropout 概率（默认 0.2；当层数>1时生效）
      - bilstm_pool: 池化方式 'mean' | 'last' | 'max'（默认 'mean'）
    """
    def __init__(self, args):
        super().__init__()
        self.enc_in    = getattr(args, 'enc_in', None)
        self.num_class = getattr(args, 'num_class', None)
        assert self.enc_in is not None and self.num_class is not None, \
            "args.enc_in 和 args.num_class 必须提供"

        hidden_dim   = getattr(args, 'lstm_hidden', 128)
        num_layers   = getattr(args, 'lstm_layers', 2)
        dropout_prob = getattr(args, 'lstm_dropout', 0.2)
        self.pooling = getattr(args, 'bilstm_pool', 'mean').lower()
        assert self.pooling in ('mean', 'last', 'max'), "bilstm_pool 必须是 mean/last/max"

        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
            bidirectional=True
        )

        out_dim = hidden_dim * 2  # 双向
        mid_dim = max(64, out_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(mid_dim, self.num_class)
        )

    @staticmethod
    def _masked_mean(x_blc, mask_bl=None, eps: float = 1e-8):
        # x_blc: [B, L, C]
        if mask_bl is None:
            return x_blc.mean(dim=1)
        w = mask_bl.float().clamp(0, 1)  # [B, L]
        denom = w.sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1]
        return (x_blc * w.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _masked_max(x_blc, mask_bl=None):
        # 将 padding 位置置为极小值再做 max
        if mask_bl is None:
            return x_blc.max(dim=1).values
        very_neg = torch.finfo(x_blc.dtype).min
        m = mask_bl.unsqueeze(-1).bool()
        x_masked = x_blc.masked_fill(~m, very_neg)
        return x_masked.max(dim=1).values

    @staticmethod
    def _gather_last(x_blc, mask_bl=None):
        # 取每个序列的“最后有效时间步”特征；若无 mask，则取最后一位
        B, L, C = x_blc.shape
        if mask_bl is None:
            idx = torch.full((B,), L - 1, device=x_blc.device, dtype=torch.long)
        else:
            lengths = mask_bl.long().sum(dim=1)              # [B]
            idx = (lengths - 1).clamp(min=0)                 # [B]
        arange_b = torch.arange(B, device=x_blc.device)
        return x_blc[arange_b, idx, :]                       # [B, C]

    def forward(self, x, padding_mask=None, x_mark=None, y_mark=None):
        """
        x: [B, L, C]
        padding_mask: [B, L] (1/True=有效, 0/False=padding)，可为 None
        """
        # LSTM 编码
        lstm_out, _ = self.lstm(x)  # [B, L, 2H]

        # 池化到序列向量
        if self.pooling == 'mean':
            feat = self._masked_mean(lstm_out, padding_mask)     # [B, 2H]
        elif self.pooling == 'max':
            feat = self._masked_max(lstm_out, padding_mask)      # [B, 2H]
        else:  # 'last'
            feat = self._gather_last(lstm_out, padding_mask)     # [B, 2H]

        # 分类头
        logits = self.classifier(feat)  # [B, num_class]
        return logits
