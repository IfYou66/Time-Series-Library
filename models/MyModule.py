import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# 1) 序列分解模块：SeriesDecomp
# ------------------------------------------------------------
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, count_include_pad=False
        )
    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        # 转 [B,C,T] 做移动平均
        trend = self.avg_pool(x.permute(0,2,1)).permute(0,2,1)  # [B, T, C]
        seasonal = x - trend
        return seasonal, trend

# ------------------------------------------------------------
# 2) 自相关函数：Auto-Correlation
# ------------------------------------------------------------
def auto_correlation(x: torch.Tensor, k: int):
    # x: [B, T, C]
    B, T, C = x.shape
    # FFT→功率谱
    Xf = torch.fft.rfft(x, dim=1)              # [B, F, C]
    P  = Xf * torch.conj(Xf)                   # [B, F, C]
    Pm = P.mean(0).mean(-1)                    # [F]
    # irFFT→自相关序列 r(τ)
    r = torch.fft.irfft(Pm, n=T, dim=0)        # [T]
    r[0] = 0
    vals, idx = torch.topk(r, k)               # 选 Top‐k 滞后
    w = F.softmax(vals, dim=0)                 # [k]
    return idx.tolist(), w                     # periods, weights

# ------------------------------------------------------------
# 3) TimesBlock（周期对齐 + 2D-Conv + 趋势残差）
# ------------------------------------------------------------
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len    = configs.seq_len
        self.pred_len   = configs.pred_len
        self.k          = configs.top_k
        self.kernel     = configs.moving_avg
        # 分解 & Inception 2D
        self.decomp     = SeriesDecomp(self.kernel)
        self.conv2d     = nn.Sequential(
            # 这里用原 Inception_Block_V1
            nn.Conv2d(configs.d_model, configs.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(configs.d_model, configs.d_model, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        B, T, C = x.shape
        # 1) 分解
        s, t = self.decomp(x)                   # seasonal, trend
        # 2) 自相关选周期
        periods, weights = auto_correlation(s, self.k)
        outs = []
        total = self.seq_len + self.pred_len
        for p, w in zip(periods, weights):
            # roll 对齐
            rolled = torch.roll(s, -p, dims=1)
            # pad & reshape→2D: [B,C,cycles,p]
            L = total
            if L % p != 0:
                pad = ( (L//p+1)*p - L )
                rolled = F.pad(rolled, (0,0,0,pad))
            cyc = rolled.shape[1] // p
            z = rolled.reshape(B, cyc, p, C).permute(0,3,1,2)  # [B,C,cyc,p]
            # 2D conv
            z = self.conv2d(z)
            # reshape回1D
            y = z.permute(0,2,3,1).reshape(B, cyc*p, C)[:,:L,:]
            outs.append(y * w)  # 直接加权
        # 聚合 & 加回趋势 + 残差
        agg = torch.stack(outs, dim=0).sum(0)      # [B,L,C]
        out = agg + t
        return out + x

# ------------------------------------------------------------
# 4) 分类模型：简单投影 + N×TimesBlock + 池化 + 分类头
# ------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 输入特征→d_model
        self.project = nn.Linear(configs.enc_in, configs.d_model)
        # 核心块
        self.blocks  = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])
        # 每层归一化
        self.norms   = nn.ModuleList([
            nn.LayerNorm(configs.d_model) for _ in range(configs.e_layers)
        ])
        # 池化和分类
        self.dropout    = nn.Dropout(configs.dropout)
        self.classifier = nn.Linear(configs.d_model, configs.num_class)

    def forward(self, x_enc, *args):
        """
        x_enc: [B, T, enc_in]
        """
        # 1) 特征投影
        x = self.project(x_enc)           # [B, T, d_model]
        # 2) N 层 TimesBlock
        for blk, ln in zip(self.blocks, self.norms):
            x = ln(blk(x))                # [B, T, d_model]
        # 3) 全局平均池化
        x = x.mean(dim=1)                 # [B, d_model]
        x = self.dropout(x)
        # 4) 分类输出
        out = self.classifier(x)          # [B, num_class]
        return out
