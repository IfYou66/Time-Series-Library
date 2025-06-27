import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """
    使用FFT检测时间序列的主要周期
    核心思想：通过频域分析找到时间序列的主要频率成分
    
    参数：
        x: 输入张量 [B, T, C] - 批次大小、时间步长、特征维度
        k: 返回前k个主要周期
    
    返回：
        period: 主要周期列表
        period_weight: 对应的权重
    """
    # [B, T, C] - 批次大小、时间步长、特征维度
    xf = torch.fft.rfft(x, dim=1)  # 对时间维度进行FFT变换
    # 通过振幅找到主要周期
    frequency_list = abs(xf).mean(0).mean(-1)  # 计算每个频率的平均振幅
    frequency_list[0] = 0  # 将直流分量（频率0）设为0
    _, top_list = torch.topk(frequency_list, k)  # 找到前k个最大振幅的频率
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list  # 计算对应的周期长度
    return period, abs(xf).mean(-1)[:, top_list]  # 返回周期和对应的权重


class TimesBlock(nn.Module):
    """
    TimesNet的核心模块：将1D时间序列转换为2D表示进行处理
    
    核心思想：
    1. 通过FFT检测时间序列的主要周期
    2. 将1D时间序列重塑为2D表示（时间-周期）
    3. 使用2D卷积捕获时间-周期维度的模式
    4. 自适应聚合不同周期的结果
    """
    
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len  # 输入序列长度
        self.pred_len = configs.pred_len  # 预测序列长度
        self.k = configs.top_k  # 使用前k个主要周期
        
        # 参数高效设计：使用Inception块进行2D卷积
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),  # 第一个Inception块
            nn.GELU(),  # 激活函数
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)  # 第二个Inception块
        )

    def forward(self, x):
        """
        TimesBlock的前向传播
        关键步骤：
        1. FFT周期检测
        2. 1D到2D转换
        3. 2D卷积处理
        4. 2D到1D转换
        5. 自适应聚合
        6. 残差连接
        """
        B, T, N = x.size()  # 批次大小、时间步长、特征维度
        
        # 1. FFT周期检测：找到时间序列的主要周期
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]  # 当前处理的周期
            
            # 2. 填充处理：确保序列长度能被周期整除
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)  # 添加填充
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            
            # 3. 1D到2D转换：重塑为(批次, 特征, 周期数, 周期长度)
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # 4. 2D卷积：从1D变化转换为2D变化处理
            out = self.conv(out)
            
            # 5. 2D到1D转换：重塑回原始格式
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        
        # 6. 堆叠不同周期的结果
        res = torch.stack(res, dim=-1)
        
        # 7. 自适应聚合：使用softmax权重聚合不同周期的结果
        period_weight = F.softmax(period_weight, dim=1)  # 归一化权重
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)  # 扩展维度
        res = torch.sum(res * period_weight, -1)  # 加权求和
        
        # 8. 残差连接：保持梯度流动
        res = res + x
        return res


class Model(nn.Module):
    """
    TimesNet主模型
    论文链接: https://openreview.net/pdf?id=ju_Uqw384Oq
    
    支持多种任务：
    - 长期预测 (long_term_forecast)
    - 短期预测 (short_term_forecast) 
    - 插值 (imputation)
    - 异常检测 (anomaly_detection)
    - 分类 (classification)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # 1. TimesBlock层：核心处理模块
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        
        # 2. 数据嵌入层：将原始特征转换为模型维度
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        # 3. 任务特定的输出层
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 预测任务：需要时间维度对齐和特征投影
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
            
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            # 插值和异常检测：只需要特征投影
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
            
        if self.task_name == 'classification':
            # 分类任务：需要展平序列并进行分类投影
            self.act = F.gelu  # 激活函数
            self.dropout = nn.Dropout(configs.dropout)  # Dropout层
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)  # 分类头

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        预测任务的前向传播
        关键步骤：
        1. 标准化（来自Non-stationary Transformer）
        2. 数据嵌入
        3. 时间维度对齐
        4. TimesBlock处理
        5. 特征投影
        6. 反标准化
        """
        # 1. 标准化：来自Non-stationary Transformer的标准化方法
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # 2. 数据嵌入：[B,T,C] - 批次、时间、特征
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # 3. 时间维度对齐：调整时间维度以匹配预测长度
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 4. TimesBlock处理：多层TimesBlock
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
            
        # 5. 特征投影：将模型维度投影到输出维度
        dec_out = self.projection(enc_out)

        # 6. 反标准化：恢复原始数据分布
        dec_out = dec_out.mul((stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add((means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        插值任务的前向传播
        特点：使用掩码进行缺失值处理
        """
        # 1. 基于掩码的标准化
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)  # 将缺失值设为0
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # 2. 数据嵌入和TimesBlock处理
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
            
        # 3. 特征投影
        dec_out = self.projection(enc_out)

        # 4. 反标准化
        dec_out = dec_out.mul((stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add((means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        异常检测任务的前向传播
        特点：不需要时间标记信息
        """
        # 1. 标准化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # 2. 数据嵌入和TimesBlock处理（不使用时间标记）
        enc_out = self.enc_embedding(x_enc, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
            
        # 3. 特征投影
        dec_out = self.projection(enc_out)

        # 4. 反标准化
        dec_out = dec_out.mul((stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add((means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        分类任务的前向传播
        关键特点：
        1. 不使用时间标记进行嵌入
        2. 使用填充掩码处理变长序列
        3. 展平序列进行全局分类
        4. 输出类别概率分布
        """
        # 1. 数据嵌入：不使用时间标记
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        
        # 2. TimesBlock处理：多层TimesBlock
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # 3. 输出处理
        # transformer编码器/解码器输出不包括非线性激活
        output = self.act(enc_out)  # 添加非线性激活
        output = self.dropout(output)  # 添加Dropout
        
        # 4. 处理填充：将填充位置的嵌入置零
        output = output * x_mark_enc.unsqueeze(-1)  # 使用填充掩码
        
        # 5. 序列展平：(batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        
        # 6. 分类投影：(batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        模型主前向传播函数
        根据任务类型调用相应的处理方法
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D] - 只返回预测部分
            
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D] - 返回完整序列
            
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D] - 返回异常检测结果
            
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N] - 返回类别概率分布
            
        return None
