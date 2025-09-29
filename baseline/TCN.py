# models/TCN.py
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# --------- 小工具：截掉因果 padding 产生的右侧多余步长 ---------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: [B, C, L + chomp_size]
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

# --------- 残差 TCN Block：dilated causal conv -> ReLU -> Dropout -> conv -> ReLU -> Dropout + Residual ---------
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        # 因果卷积需要在左侧 pad (kernel_size-1)*dilation
        pad = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           padding=pad, dilation=dilation))
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           padding=pad, dilation=dilation))
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(inplace=True)

        # 初始化稍微保守一点
        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x):
        # x: [B, C, L]
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # 残差 + ReLU

# --------- 堆叠若干 TCN Block 形成 TCN 主干 ---------
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        """
        num_inputs:  输入通道（特征维）
        num_channels: list[int]，每一层的通道数（长度 = levels）
        """
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i  # 指数膨胀
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, L]
        return self.network(x)  # [B, C_last, L]

# --------- 顶层分类模型（供 Exp_Classification 调用）---------
class Model(nn.Module):
    """
    兼容 Exp_Classification：
      outputs = model(batch_x, padding_mask, None, None)
    超参读取（有默认值）：
      - args.enc_in:    输入维度（必需）
      - args.num_class: 类别数（必需）
      - args.tcn_hidden:     第一层通道数（默认 128）
      - args.tcn_levels:     残差层数（默认 4）
      - args.tcn_kernel:     卷积核大小（默认 3）
      - args.tcn_dropout:    Dropout（默认 0.2）
    """
    def __init__(self, args):
        super().__init__()
        self.enc_in    = getattr(args, 'enc_in', None)
        self.num_class = getattr(args, 'num_class', None)
        assert self.enc_in is not None and self.num_class is not None, \
            "args.enc_in 与 args.num_class 不能为空"

        hidden      = getattr(args, 'tcn_hidden', 128)
        levels      = getattr(args, 'tcn_levels', 4)
        ksize       = getattr(args, 'tcn_kernel', 3)
        dropout     = getattr(args, 'tcn_dropout', 0.2)

        # 主干：TCN（空洞因果卷积 + 残差）
        # 通道配置：enc_in -> hidden -> ... -> hidden
        num_channels = [hidden] * levels
        self.tcn = TemporalConvNet(num_inputs=self.enc_in,
                                   num_channels=num_channels,
                                   kernel_size=ksize,
                                   dropout=dropout)

        # 分类头：掩码全局平均池化（沿时间） + 全连接
        self.classifier = nn.Sequential(
            nn.Linear(hidden, max(64, hidden // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(max(64, hidden // 2), self.num_class)
        )

    @staticmethod
    def _masked_mean_pool(x_bcl, mask_bl=None, eps: float = 1e-8):
        """
        x_bcl: [B, C, L]
        mask_bl: [B, L] (1=有效/True, 0=padding/False)；None 则普通均值
        返回: [B, C]
        """
        if mask_bl is None:
            return x_bcl.mean(dim=-1)

        # 统一成 float 权重
        if mask_bl.dtype != torch.float32 and mask_bl.dtype != torch.float64:
            mask_bl = mask_bl.float()
        mask_bl = mask_bl.clamp(min=0.0, max=1.0)

        # 加一个很小的数避免除零
        denom = mask_bl.sum(dim=-1, keepdim=True).clamp_min(eps)  # [B, 1]
        # 扩展到通道维
        w = mask_bl.unsqueeze(1)  # [B, 1, L]
        num = (x_bcl * w).sum(dim=-1)  # [B, C]
        return num / denom  # [B, C]

    def forward(self, x, padding_mask=None, x_mark=None, y_mark=None):
        """
        x: [B, L, C]
        padding_mask: [B, L]  (1/True=有效，0/False=padding)
        """
        # 1) 转成 Conv1d 需要的 [B, C, L]
        x = x.transpose(1, 2).contiguous()

        # 2) TCN 主干
        feat = self.tcn(x)  # [B, hidden, L]

        # 3) 掩码全局平均池化（沿时间）
        pooled = self._masked_mean_pool(feat, padding_mask)  # [B, hidden]

        # 4) 分类头
        logits = self.classifier(pooled)  # [B, num_class]
        return logits
