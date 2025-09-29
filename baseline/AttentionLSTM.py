import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """
    简单的时间维 Attention：对 LSTM 输出加权求和
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out, mask=None):
        """
        lstm_out: [B, L, H]
        mask: [B, L] (1=有效, 0=padding)，可选
        """
        # [B, L, 1]
        scores = self.attn(torch.tanh(lstm_out))

        if mask is not None:
            # mask=0 的位置赋 -inf，避免影响 softmax
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))

        # softmax → [B, L, 1]
        attn_weights = torch.softmax(scores, dim=1)

        # 加权求和 → [B, H]
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return context, attn_weights


class Model(nn.Module):
    """
    Attention-LSTM 分类模型
    兼容 Exp_Classification：
      outputs = model(batch_x, padding_mask, None, None)
    """
    def __init__(self, args):
        super().__init__()
        self.enc_in = getattr(args, 'enc_in', None)       # 输入维度
        self.num_class = getattr(args, 'num_class', None) # 类别数
        assert self.enc_in is not None and self.num_class is not None, \
            "args.enc_in 和 args.num_class 必须指定"

        hidden_dim = getattr(args, 'lstm_hidden', 128)    # LSTM 隐藏维
        num_layers = getattr(args, 'lstm_layers', 2)      # LSTM 层数
        dropout = getattr(args, 'lstm_dropout', 0.2)

        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = AttentionLayer(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_class)
        )

    def forward(self, x, padding_mask=None, x_mark=None, y_mark=None):
        """
        x: [B, L, C]
        padding_mask: [B, L] (1=有效, 0=padding)，可选
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # [B, L, H]

        # Attention pooling
        context, attn_weights = self.attention(lstm_out, mask=padding_mask)  # [B, H]

        # 分类
        logits = self.classifier(context)  # [B, num_class]
        return logits
