import torch
import math
import torch.nn as nn
from layer import TransformerBlock
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GCNConv
from utils import is_edge_in_edge_index


class PretrainModel(nn.Module):
    """
    基于 Transformer 的预训练模型，用于图表示学习。
    支持多跳特征聚合和图全局池化操作。
    """

    def __init__(self, input_dim, config):
        """
        初始化模型。

        参数:
        - input_dim (int): 输入特征的维度。
        - config (Namespace): 模型配置，包括以下超参数:
            - hidden_dim (int): 隐藏层特征维度。
            - hops (int): 聚合的跳数。
            - n_layers (int): Transformer 层数。
            - n_heads (int): Transformer 多头注意力头数。
            - dropout (float): Dropout 比例。
            - attention_dropout (float): 注意力模块的 Dropout 比例。
            - readout (str): 池化方式 ("sum", "mean", "max")。
            - device (torch.device): 计算设备 (CPU 或 GPU)。
            - alpha (float): 链接损失权重。
        """
        super().__init__()
        self.input_dim = input_dim  # 输入特征维度
        self.config = config  # 模型配置

        # 线性变换，将输入特征映射到隐藏特征空间
        self.Linear1 = nn.Linear(input_dim, self.config.hidden_dim)

        # Transformer 编码器，用于多跳特征聚合
        self.encoder = TransformerBlock(
            hops=config.hops,
            input_dim=input_dim,
            n_layers=config.n_layers,
            num_heads=config.n_heads,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout,
            attention_dropout_rate=config.attention_dropout
        )

        # 根据配置选择全局池化方式
        if config.readout == "sum":
            self.readout = global_add_pool
        elif config.readout == "mean":
            self.readout = global_mean_pool
        elif config.readout == "max":
            self.readout = global_max_pool
        else:
            raise ValueError("Invalid pooling type.")  # 如果无效，抛出错误

        # 边界损失，用于对比损失计算
        self.marginloss = nn.MarginRankingLoss(0.5)

    def forward(self, x):
        """
        模型前向传播。

        参数:
        - x (torch.Tensor): 输入特征张量，形状为 (batch_size, features_dim)。

        返回:
        - node_tensor (torch.Tensor): 节点的嵌入表示，形状为 (batch_size, hidden_dim)。
        - neighbor_tensor (torch.Tensor): 节点邻居的嵌入表示，形状为 (batch_size, hidden_dim)。
        """
        # 使用 Transformer 编码器提取节点和邻居的嵌入特征
        node_tensor, neighbor_tensor = self.encoder(x)  # (batch_size, 1, hidden_dim), (batch_size, hops, hidden_dim)

        # 对邻居特征使用全局池化
        neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device))  # (batch_size, 1, hidden_dim)

        # 去掉多余的维度并返回
        return node_tensor.squeeze(), neighbor_tensor.squeeze()

    def contrastive_link_loss(self, node_tensor, neighbor_tensor, adj_, minus_adj):
        """
        计算对比损失 (Contrastive Loss)，结合对比学习和图链接预测任务。

        参数:
        - node_tensor (torch.Tensor): 节点的嵌入表示，形状为 (N, d)。
        - neighbor_tensor (torch.Tensor): 节点邻居的嵌入表示，形状为 (N, d)。
        - adj_ (torch.Tensor): 原始邻接矩阵 (稀疏矩阵)，表示图的连接关系。
        - minus_adj (torch.Tensor): 补图邻接矩阵 (稀疏矩阵)，表示不存在连接的节点对。

        返回:
        - TotalLoss (torch.Tensor): 总损失值，包含对比损失和链接损失。
        """
        # 1. 生成随机排列的节点和邻居张量
        shuf_index = torch.randperm(node_tensor.shape[0])  # 随机打乱索引
        node_tensor_shuf = node_tensor[shuf_index]  # 随机排列的节点张量
        neighbor_tensor_shuf = neighbor_tensor[shuf_index]  # 随机排列的邻居张量

        # 2. 计算节点对的相似性分数 (logits)
        logits_aa = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor, dim=-1))  # 原始节点-邻居
        logits_bb = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor_shuf, dim=-1))  # 随机节点-随机邻居
        logits_ab = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor_shuf, dim=-1))  # 原始节点-随机邻居
        logits_ba = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor, dim=-1))  # 随机节点-原始邻居

        # 3. 计算对比损失
        TotalLoss = 0.0  # 初始化总损失
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)  # 全 1 张量
        TotalLoss += self.marginloss(logits_aa, logits_ba, ones)  # 边界损失: 原始 vs 随机
        TotalLoss += self.marginloss(logits_bb, logits_ab, ones)  # 边界损失: 随机 vs 原始

        # 4. 计算链接损失
        pairwise_similary = torch.mm(node_tensor, node_tensor.t())  # 节点嵌入的成对相似性
        link_loss = minus_adj.multiply(pairwise_similary) - adj_.multiply(pairwise_similary)  # 链接损失
        link_loss = torch.sum(link_loss) / (adj_.shape[0] * adj_.shape[0])  # 标准化损失

        # 加权链接损失并添加到总损失中
        TotalLoss += self.config.alpha * link_loss

        # 返回总损失
        return TotalLoss
