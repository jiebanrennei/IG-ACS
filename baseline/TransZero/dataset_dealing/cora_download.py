# 导入相关库
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 选择设备为GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"device : {device}")

# 函数：edge_index_to_sparse_coo
# 功能：将边索引(edge_index)转换为稀疏COO格式的稀疏矩阵
# 输入：edge_index (Tensor): 一个形状为 [2, num_edges] 的张量，表示图的边的源节点和目标节点索引
# 输出：edge_index_sparse (SparseTensor): 一个稀疏张量，表示图的边以COO格式存储
def edge_index_to_sparse_coo(edge_index):
    # 将边的源节点和目标节点的索引转换为long类型
    row = edge_index[0].long()
    col = edge_index[1].long()

    # 构建稀疏矩阵的形状, num_nodes是图中节点的最大编号+1（也可以代表节点的数量）
    num_nodes = torch.max(edge_index) + 1
    size = (num_nodes.item(), num_nodes.item())     # 稀疏邻接矩阵的形状

    # 构建稀疏矩阵,所有边的权重为1
    values = torch.ones_like(row)       # 创建一个与row完全相同的张量，并且值全是1
    edge_index_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), values, size)

    return edge_index_sparse

dataset_str = 'cora'    # 设置数据集名称

# 从PyG库加载Cora数据集
# 输入：无
# 输出：dataset (Planetoid): 一个包含Cora数据集的对象
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='./data/cora', name='cora')

# 获取图数据对象，包含图的节点特征 (x), 边索引 (edge_index), 标签 (y) 等信息
# 输入：dataset[0] 返回的数据对象
# 输出：graph (Data): 图数据对象，包含图的节点特征、边索引、标签等信息
graph = dataset[0]

# print(graph.x, graph.edge_index, graph.y)
# 打印边
print(graph.edge_index)

# 将转换后的数据保存到本地文件
# 输入：graph.edge_index, graph.x, graph.y: 图的边索引、节点特征和节点标签
# 输出：保存的文件，包含edge_index（稀疏矩阵）、节点特征和节点标签，格式为 LongTensor
torch.save([edge_index_to_sparse_coo(graph.edge_index).type(torch.LongTensor), graph.x.type(torch.LongTensor), graph.y.type(torch.LongTensor)], "./dataset/"+dataset_str+"_pyg.pt")

