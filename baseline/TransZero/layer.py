import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# 初始化模型参数
def init_params(module, n_layers):
    """
    初始化模型的线性层和嵌入层的权重
    使用正态分布初始化权重，偏置初始化为0
    """
    if isinstance(module, nn.Linear):
        # 线性层的权重使用均值为0，标准差为 0.02 / sqrt(n_layers) 的正态分布初始化
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            # 偏置初始化为0
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        # 嵌入层的权重使用均值为0，标准差为 0.02 的正态分布初始化
        module.weight.data.normal_(mean=0.0, std=0.02)


# GELU 激活函数
def gelu(x):
    """
    GELU激活函数，参考文章：https://arxiv.org/abs/1606.08415
    公式为: 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


# Transformer 块，包含自注意力机制和前馈网络
class TransformerBlock(nn.Module):
    def __init__(
            self,
            hops,  # 水平跳数，确定序列长度
            input_dim,  # 输入特征维度
            n_layers=6,  # Transformer的层数，默认6层
            num_heads=8,  # 多头注意力机制的头数，默认8
            hidden_dim=64,  # 隐藏层的维度，默认64
            dropout_rate=0.0,  # Dropout比率，默认0.0
            attention_dropout_rate=0.1  # 注意力层的Dropout比率，默认0.1
    ):
        super().__init__()

        # 初始化模型参数
        self.seq_len = hops + 1  # 序列长度，包含目标节点和邻居节点
        self.input_dim = input_dim  # 输入特征维度
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.ffn_dim = 2 * hidden_dim  # 前馈网络的维度，通常为隐藏层维度的2倍
        self.num_heads = num_heads  # 多头注意力机制的头数
        self.n_layers = n_layers  # Transformer的层数
        self.dropout_rate = dropout_rate  # Dropout比率
        self.attention_dropout_rate = attention_dropout_rate  # 注意力层的Dropout比率

        # 定义输入到隐藏层的线性变换，输入维度转换为隐藏层维度
        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)

        # 创建多个 EncoderLayer，构成Transformer Block
        encoders = [
            EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
            for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)  # 使用ModuleList存储多个EncoderLayer

        # 最后的LayerNorm层，用于稳定训练过程
        self.final_ln = nn.LayerNorm(hidden_dim)

        # 输出投影层，将隐藏层维度减半
        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))

        # 用于计算目标节点与邻居节点之间注意力权重的线性层
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)

        # 初始化缩放因子，用于调整模型输出的大小
        self.scaling = nn.Parameter(torch.ones(1) * 0.5)

        # 应用参数初始化函数
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        """
        前向传播函数，计算目标节点与邻居节点之间的加权表示
        参数:
            batched_data: 输入的批量数据，形状为 (batch_size, seq_len, input_dim)
            - batch_size: 批量大小
            - seq_len: 序列长度（包括目标节点和邻居节点）
            - input_dim: 每个节点的输入特征维度

        输出:
            node_tensor: 目标节点的表示，形状为 (batch_size, hidden_dim)
            neighbor_tensor: 加权后的邻居节点的表示，形状为 (batch_size, hidden_dim)
        """

        # 输入数据通过嵌入层（线性变换）转换为隐藏层表示
        # 输入维度 (batch_size, seq_len, input_dim) -> 输出维度 (batch_size, seq_len, hidden_dim)
        tensor = self.att_embeddings_nope(batched_data)

        # 依次通过每一层的EncoderLayer
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)

        # 对最终输出进行LayerNorm处理
        output = self.final_ln(tensor)  # 输出维度 (batch_size, seq_len, hidden_dim)

        # 提取目标节点和邻居节点的张量
        # 目标节点是序列中的第一个节点，邻居节点是其余节点
        target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)  # 目标节点的表示，形状 (batch_size, seq_len-1, hidden_dim)
        split_tensor = torch.split(output, [1, self.seq_len - 1], dim=1)

        node_tensor = split_tensor[0]  # 目标节点的表示，形状 (batch_size, 1, hidden_dim)
        neighbor_tensor = split_tensor[1]  # 邻居节点的表示，形状 (batch_size, seq_len-1, hidden_dim)

        # 计算目标节点和邻居节点的注意力权重
        # 连接目标节点和邻居节点的表示后输入到注意力层，形状 (batch_size, seq_len-1, 2*hidden_dim) -> 输出注意力权重 (batch_size, seq_len-1, 1)
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))

        # 对注意力权重进行softmax归一化，确保所有注意力权重和为1
        layer_atten = F.softmax(layer_atten, dim=1)  # 形状 (batch_size, seq_len-1, 1)

        # 根据注意力权重对邻居节点表示进行加权
        neighbor_tensor = neighbor_tensor * layer_atten  # 形状 (batch_size, seq_len-1, hidden_dim)

        # 返回目标节点和加权后的邻居节点表示
        return node_tensor, neighbor_tensor  # 目标节点: (batch_size, 1, hidden_dim), 邻居节点: (batch_size, seq_len-1, hidden_dim)



# 前馈网络层
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        """
        前馈神经网络，包含两个全连接层
        """
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)  # 第一层
        self.gelu = nn.GELU()  # GELU激活函数
        self.layer2 = nn.Linear(ffn_size, hidden_size)  # 第二层

    def forward(self, x):
        """
        前向传播
        """
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


# 多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        """
        多头自注意力机制
        """
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads  # 每个头的维度
        self.scale = self.att_size ** -0.5  # 缩放因子

        # 初始化查询、键、值的线性变换
        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)

        # Dropout层
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        # 输出层
        self.output_layer = nn.Linear(num_heads * self.att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        """
        前向传播
        q: 查询张量
        k: 键张量
        v: 值张量
        attn_bias: 可选的注意力偏置
        """
        orig_q_size = q.size()  # 保存原始的查询张量大小

        d_k = self.att_size  # 键的维度
        d_v = self.att_size  # 值的维度
        batch_size = q.size(0)  # 批大小

        # 计算查询、键、值的线性变换
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        # 转置以适应计算
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        # 缩放点积注意力
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]

        if attn_bias is not None:
            x = x + attn_bias  # 加上偏置

        x = torch.softmax(x, dim=3)  # 对k_len维度进行softmax
        x = self.att_dropout(x)  # Dropout
        x = x.matmul(v)  # 乘以值张量

        # 还原维度
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        # 输出层变换
        x = self.output_layer(x)

        assert x.size() == orig_q_size  # 确保输出尺寸与输入相同
        return x


# Transformer 的 Encoder 层，包含自注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        """
        初始化 Transformer Encoder 层，包括自注意力和前馈神经网络模块。

        参数:
            hidden_size: 隐藏层的维度。
            ffn_size: 前馈神经网络的大小，通常为2倍的隐藏层维度。
            dropout_rate: Dropout比率，用于避免过拟合。
            attention_dropout_rate: 注意力层的Dropout比率。
            num_heads: 多头注意力机制的头数。
        """
        super(EncoderLayer, self).__init__()

        # 自注意力层
        self.self_attention_norm = nn.LayerNorm(hidden_size)  # 自注意力层的LayerNorm
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)  # 多头自注意力层
        self.self_attention_dropout = nn.Dropout(dropout_rate)  # Dropout层

        # 前馈神经网络层
        self.ffn_norm = nn.LayerNorm(hidden_size)  # 前馈神经网络的LayerNorm
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)  # 前馈神经网络
        self.ffn_dropout = nn.Dropout(dropout_rate)  # Dropout层

    def forward(self, x, attn_bias=None):
        """
        前向传播函数，处理输入的数据并返回经过自注意力和前馈神经网络层处理后的输出。

        参数:
            x: 输入数据，形状为 (batch_size, seq_len, hidden_size)
                - batch_size: 批量大小
                - seq_len: 序列长度
                - hidden_size: 每个节点的特征维度（通常为Transformer隐藏层的维度）
            attn_bias: 注意力机制中的偏置（可选），形状为 (batch_size, num_heads, seq_len, seq_len)

        输出:
            x: 输出数据，形状为 (batch_size, seq_len, hidden_size)
                - 经过自注意力和前馈神经网络层处理后的结果
        """

        # 自注意力层：
        # 1. 对输入x进行LayerNorm归一化
        y = self.self_attention_norm(x)  # 形状保持不变 (batch_size, seq_len, hidden_size)
        # 2. 将归一化后的输入传入多头自注意力层
        y = self.self_attention(y, y, y, attn_bias)  # 形状保持不变 (batch_size, seq_len, hidden_size)
        # 3. 经过Dropout层，防止过拟合
        y = self.self_attention_dropout(y)  # 形状保持不变 (batch_size, seq_len, hidden_size)
        # 4. 输入x与注意力层的输出y相加（残差连接）
        x = x + y  # 形状保持不变 (batch_size, seq_len, hidden_size)

        # 前馈神经网络层：
        # 1. 对输入x进行LayerNorm归一化
        y = self.ffn_norm(x)  # 形状保持不变 (batch_size, seq_len, hidden_size)
        # 2. 将归一化后的输入传入前馈神经网络层
        y = self.ffn(y)  # 形状保持不变 (batch_size, seq_len, hidden_size)
        # 3. 经过Dropout层
        y = self.ffn_dropout(y)  # 形状保持不变 (batch_size, seq_len, hidden_size)
        # 4. 输入x与前馈网络的输出y相加（残差连接）
        x = x + y  # 形状保持不变 (batch_size, seq_len, hidden_size)

        return x  # 返回处理后的结果 (batch_size, seq_len, hidden_size)


class TransformerModel(nn.Module):
    def __init__(
            self,
            hops,  # 水平跳数，用于定义输入序列的长度
            n_class,  # 输出类别数
            input_dim,  # 输入特征的维度
            pe_dim,  # 位置编码维度
            n_layers=6,  # Transformer中的层数，默认6层
            num_heads=8,  # 多头注意力机制中的头数，默认8
            hidden_dim=64,  # 隐藏层维度，默认64
            ffn_dim=64,  # 前馈网络维度，默认64
            dropout_rate=0.0,  # Dropout比率，默认0.0
            attention_dropout_rate=0.1  # 注意力机制中的Dropout比率，默认0.1
    ):
        super().__init__()

        # 初始化模型的各个超参数
        self.seq_len = hops + 1  # 序列长度，hops为跳数，加1是因为需要包含目标节点
        self.pe_dim = pe_dim  # 位置编码的维度
        self.input_dim = input_dim  # 输入特征维度
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.ffn_dim = 2 * hidden_dim  # 前馈网络的维度，通常设置为隐藏层维度的2倍
        self.num_heads = num_heads  # 多头注意力机制的头数

        # 层数和类别数
        self.n_layers = n_layers  # Transformer的层数
        self.n_class = n_class  # 输出类别数

        # Dropout率，用于避免过拟合
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate  # 注意力机制中的Dropout率

        # 输入到隐藏层的线性变换
        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)

        # 定义Transformer的多个Encoder层
        encoders = [
            EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
            for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)  # 将所有Encoder层放在一个ModuleList中，方便批量处理

        # 最后的LayerNorm，帮助稳定训练
        self.final_ln = nn.LayerNorm(hidden_dim)

        # 输出投影层，将隐藏层的维度减半
        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))

        # 用于计算目标节点和邻居节点之间注意力权重的层
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)

        # 用于输出分类结果的全连接层
        self.Linear1 = nn.Linear(int(self.hidden_dim / 2), self.n_class)

        # 定义一个缩放因子用于调整模型的输出
        self.scaling = nn.Parameter(torch.ones(1) * 0.5)

        # 初始化模型中的参数
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        """
        前向传播函数
        batched_data: 输入的批量数据
        """

        # 输入数据通过嵌入层转换为隐藏层表示
        tensor = self.att_embeddings_nope(batched_data)

        # Transformer编码器进行处理，每层依次处理
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)

        # 对输出进行LayerNorm
        output = self.final_ln(tensor)

        # 获取目标节点和邻居节点
        target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        split_tensor = torch.split(output, [1, self.seq_len - 1], dim=1)

        node_tensor = split_tensor[0]  # 目标节点
        neighbor_tensor = split_tensor[1]  # 邻居节点

        # 计算目标节点与邻居节点的注意力权重
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))

        # 对注意力权重进行softmax归一化
        layer_atten = F.softmax(layer_atten, dim=1)

        # 根据注意力权重加权邻居节点的表示
        neighbor_tensor = neighbor_tensor * layer_atten

        # 对加权后的邻居节点表示进行求和，得到邻居的聚合表示
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)

        # 将目标节点与邻居的聚合表示进行相加
        output = (node_tensor + neighbor_tensor).squeeze()

        # 通过一个全连接层进行最终分类，ReLU激活函数增加非线性
        output = self.Linear1(torch.relu(self.out_proj(output)))

        # 使用log_softmax计算类别概率分布
        return torch.log_softmax(output, dim=1)
