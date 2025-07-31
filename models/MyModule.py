import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k):
    """
    使用FFT检测时间序列的主要周期
    """
    xf = torch.fft.rfft(x, dim=1)  # [B, ⌊T/2⌋+1, C]
    frequency_list = xf.abs().mean(0).mean(-1)  # [⌊T/2⌋+1]
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)  # [k]
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list  # [k]
    return period, xf.abs().mean(-1)[:, top_list]  # ([k], [B, k])



class TimesBlock(nn.Module):
    """
    多尺度 TimesNet 核心模块。并行多组 top-k → concat → 1x1 融合 → SE 注意力 → 残差。
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # 支持多尺度 top-k
        self.multi_k = getattr(configs, 'multi_k', [configs.top_k])

        # 2D 多尺度卷积路径
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
        # 1x1 卷积融合多尺度
        in_channels = configs.d_model * len(self.multi_k)
        self.fusion_conv = nn.Conv1d(in_channels, configs.d_model, kernel_size=1)

        # SE 通道注意力
        C = configs.d_model
        r = getattr(configs, 'se_ratio', 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, C, 1]
            nn.Conv1d(C, C // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(C // r, C, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, C_in = x.size()
        L = self.seq_len + self.pred_len

        multi_outs = []
        for k in self.multi_k:
            # 1) 按 k 检测周期并加权聚合
            period_list, period_weight = FFT_for_Period(x, k)
            scale_outs = []
            for i in range(k):
                p = int(period_list[i])
                # 填充
                if L % p != 0:
                    Lp = ((L // p) + 1) * p
                    pad = torch.zeros(B, Lp - L, C_in, device=x.device)
                    xi = torch.cat([x, pad], dim=1)
                else:
                    Lp = L
                    xi = x
                # 1D→2D
                xi = xi.reshape(B, Lp // p, p, C_in).permute(0, 3, 1, 2)
                xi = self.conv(xi)  # [B, d_model, Lp//p, p]
                # 2D→1D
                xi = xi.permute(0, 2, 3, 1).reshape(B, -1, C_in)
                scale_outs.append(xi[:, :L, :])  # [B, L, C_in]
            # stack & period-weighted sum
            so = torch.stack(scale_outs, dim=-1)  # [B, L, C_in, k]
            w = F.softmax(period_weight, dim=1)  # [B, k]
            w = w.unsqueeze(1).unsqueeze(1).repeat(1, T, C_in, 1)
            so = (so * w).sum(dim=-1)  # [B, L, C_in]
            multi_outs.append(so)

        # 2) 多尺度 concat 并 1x1 融合
        out = torch.cat(multi_outs, dim=2)  # [B, T, C_in * len(multi_k)]
        out = out.permute(0, 2, 1)  # [B, C_concat, T]
        out = self.fusion_conv(out)  # [B, C_in, T]
        out = out.permute(0, 2, 1)  # [B, T, C_in]

        # 3) SE 通道注意力
        out_ca = out.permute(0, 2, 1)  # [B, C_in, T]
        se_w = self.se(out_ca)  # [B, C_in, 1]
        out_ca = out_ca * se_w  # [B, C_in, T]
        out = out_ca.permute(0, 2, 1)  # [B, T, C_in]

        # 4) 残差连接
        return out + x


class Model(nn.Module):
    """
    TimesNet主模型
    支持：long_term_forecast, short_term_forecast,
          imputation, anomaly_detection, classification
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # 1. 核心TimesBlock堆叠
        self.model = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])

        # 2. 数据嵌入
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq,
            configs.dropout
        )
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 3. 任务专属Head
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len
            )
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True
            )

        if self.task_name in ('imputation', 'anomaly_detection'):
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True
            )

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len,
                configs.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 与原版完全相同
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for blk in self.model:
            enc_out = self.layer_norm(blk(enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # 与原版完全相同
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means).masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for blk in self.model:
            enc_out = self.layer_norm(blk(enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # 与原版完全相同
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, None)
        for blk in self.model:
            enc_out = self.layer_norm(blk(enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # 与原版完全相同
        enc_out = self.enc_embedding(x_enc, None)
        for blk in self.model:
            enc_out = self.layer_norm(blk(enc_out))
        out = self.act(enc_out)
        out = self.dropout(out)
        out = out * x_mark_enc.unsqueeze(-1)
        out = out.reshape(out.size(0), -1)
        out = self.projection(out)
        return out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
