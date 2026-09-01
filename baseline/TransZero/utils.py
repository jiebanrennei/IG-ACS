import argparse
import torch
import scipy.sparse as sp
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl
import scipy.sparse as sp
import numpy as np
import networkx as nx
from numpy import *
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score
from sklearn.metrics import precision_score, recall_score, f1_score
import igraph as ig
from typing import Tuple

# Training settings
def parse_args():
    """
       生成并配置命令行参数解析器，用于模型训练和配置的超参数设置

       返回:
           argparse.Namespace: 包含所有解析参数的对象
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # --------------------------
    # 主要运行参数
    # --------------------------
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Choose from {pubmed}')
    parser.add_argument('--device', type=int, default=1, 
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed.')

    # model parameters

    # --------------------------
    # 模型架构参数
    # --------------------------
    parser.add_argument('--hops', type=int, default=7,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--pe_dim', type=int, default=15,
                        help='position embedding size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    parser.add_argument('--readout', type=str, default="mean")
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='the value the balance the loss.')

    # training parameters
    # --------------------------
    # 训练超参数
    # --------------------------
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    parser.add_argument('--group_epoch_gap', type=int, default=20,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay    L2正则化系数')
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')

    # --------------------------
    # 模型保存配置
    # --------------------------
    parser.add_argument('--save_path', type=str, default='./model/',
                        help='模型保存根目录')
    parser.add_argument('--model_name', type=str, default='cora',
                        help='保存模型时使用的前缀名称')
    parser.add_argument('--embedding_path', type=str, default='./pretrain_result/',
                        help='预训练嵌入向量的保存路径')

    return parser.parse_args()


def laplacian_positional_encoding(g, pos_enc_dim):
    """
    计算图的拉普拉斯位置编码（Laplacian Positional Encoding），即图的拉普拉斯算子特征向量。

    参数：
        g (DGLGraph): DGL 图对象，包含图的结构（节点和边的信息）。
        pos_enc_dim (int): 位置编码的维度，表示需要计算多少个特征向量作为位置编码。

    返回：
        lap_pos_enc (torch.Tensor): 拉普拉斯位置编码矩阵，形状为 [num_nodes, pos_enc_dim]，即每个节点的位置信息。
    """

    # 获取图的邻接矩阵（采用稀疏矩阵格式 CSR）
    # A 是图的邻接矩阵，形状为 [num_nodes, num_nodes]
    A = g.adjacency_matrix(scipy_fmt='csr')

    # 获取节点的度，并计算归一化矩阵 N，N 是一个对角矩阵，形状为 [num_nodes, num_nodes]
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)

    # 计算图的拉普拉斯矩阵 L = I - N * A * N，L 的形状为 [num_nodes, num_nodes]
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # 计算拉普拉斯矩阵 L 的特征值和特征向量（使用 scipy 的 eigs 函数）
    # EigVec 是形状为 [num_nodes, k] 的矩阵，包含 k 个特征向量
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)  # 计算 pos_enc_dim+1 个特征向量

    # 对特征值进行排序，选择按升序排列的特征向量
    EigVec = EigVec[:, EigVal.argsort()]  # 按特征值的升序排序特征向量

    # 选择从第二个特征向量开始的 pos_enc_dim 个特征向量作为位置编码
    # 因为第一个特征向量是常数向量，通常不作为位置编码使用
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()  # 形状为 [num_nodes, pos_enc_dim]

    return lap_pos_enc


import torch  # 导入 PyTorch 库，用于深度学习


def re_features(adj, features, K):
    """
    根据图的邻接矩阵传播节点特征，并返回传播后的特征矩阵。

    参数：
        adj (torch.Tensor): 图的邻接矩阵，形状为 [num_nodes, num_nodes]，表示节点之间的连接关系。
        features (torch.Tensor): 图节点的特征矩阵，形状为 [num_nodes, feature_dim]，其中 feature_dim 表示每个节点的特征维度。
        K (int): 传播的步数，表示节点特征通过邻接矩阵传播的次数。hop数

    返回：
        nodes_features (torch.Tensor): 传播后的特征矩阵，形状为 [num_nodes, K+1, feature_dim]，包含传播的 K+1 个时刻的节点特征。
    """

    # 初始化存储每个节点传播后特征的矩阵，shape = (num_nodes, 1, K+1, feature_dim)
    nodes_features = torch.empty(features.shape[0], 1, K + 1, features.shape[1])

    # 将每个节点的初始特征存储到节点特征矩阵的第一个位置 0-hop
    for i in range(features.shape[0]):
        nodes_features[i, 0, 0, :] = features[i]

    # 创建一个与 features 形状相同的零矩阵，目的是用于后续传播
    x = features + torch.zeros_like(features)

    # 进行 K 次传播
    for i in range(K):
        # 通过邻接矩阵进行特征传播，即将特征矩阵乘以邻接矩阵
        x = torch.matmul(adj, x)

        # 将传播后的特征保存到 nodes_features 中的第 i+1 个位置
        for index in range(features.shape[0]):
            nodes_features[index, 0, i + 1, :] = x[index]

    # 删除不必要的维度，返回最终的传播后的特征矩阵，shape = (num_nodes, K+1, feature_dim)
    nodes_features = nodes_features.squeeze()

    return nodes_features


def conductance_hop(adj, max_khop):
    """
    计算图的传导性（conductance）跳数（hop）的分布，基于图的邻接矩阵。

    参数:
        adj (torch.Tensor): 图的邻接矩阵，形状为 (N, N)，其中 N 是节点数。
        max_khop (int): 最大跳数（hop）的值，表示计算传导性时考虑的最大跳数。

    返回:
        results (torch.Tensor): 计算出的传导性结果，形状为 (N, max_khop+1)，表示每个节点在不同跳数下的传导性分布。
    """

    adj = adj.to(dtype=torch.float)  # 将邻接矩阵转换为浮点型，确保矩阵运算时的精度
    adj_current_hop = adj  # 当前跳数的邻接矩阵初始化为原始邻接矩阵

    # 初始化结果张量，形状为 (max_khop+1, N)，其中 N 是节点数，max_khop 是最大跳数
    results = torch.zeros((max_khop + 1, adj.shape[0]))

    # 循环遍历每个跳数（从 0 到 max_khop）
    for hop in range(max_khop + 1):
        adj_current_hop = torch.matmul(adj_current_hop, adj)  # 计算当前跳数的邻接矩阵（即跳数为 hop 的邻接矩阵）

        # 计算当前跳数邻接矩阵的节点度（每个节点的连接数）
        degree = torch.sum(adj_current_hop, dim=0)  # 每个节点的度，形状为 (N,)

        # 对当前跳数邻接矩阵取符号，得到一个表示边是否存在的矩阵
        adj_current_hop_sign = torch.sign(adj_current_hop)  # 符号矩阵，形状为 (N, N)

        # 计算符号矩阵的度，表示存在边的节点数
        degree_1 = torch.sum(adj_current_hop_sign, dim=0)  # 每个节点的符号度，形状为 (N,)

        # 计算当前跳数节点的传导性并将其存入结果矩阵
        results[hop] = (degree - degree_1).to_dense().reshape(1, -1)  # 每个节点在当前跳数的传导性

        hop += 1  # 增加跳数（此行代码实际上不需要，因为 for 循环已自动递增）

    # 将结果矩阵转置，形状变为 (N, max_khop+1)，N 是节点数
    results = results.T

    # 对每个节点找到最大传导性对应的跳数
    max_indices = torch.argmax(results, dim=1)  # max_indices 的形状为 (N,)，表示每个节点最大传导性对应的跳数

    # 根据最大传导性的跳数调整传导性矩阵，保证在最大传导性之后的跳数都为 0
    for i in range(results.shape[0]):  # 遍历每个节点
        for j in range(results.shape[1]):  # 遍历每个跳数
            # 如果跳数大于最大传导性对应的跳数，并且最大跳数不为 0，将该位置设为 0
            if j > max_indices[i] and max_indices[i] != 0:
                results[i][j] = 0
            else:
                results[i][j] = 1  # 否则，设置为 1（表示存在传导性）

    # 如果 hop==1，即只有一个跳数时，将所有结果设为 1
    if hop == 1:
        results = torch.ones((max_khop + 1, adj.shape[0]))

    return results  # 返回最终计算的传导性矩阵，形状为 (N, max_khop+1)


# def f1_score_calculation(y_pred, y_true):
#     if len(y_pred.shape) == 1:
#         y_pred = y_pred.reshape(1, -1)
#         y_true = y_true.reshape(1, -1)
#     F1 = []
#     for i in range(y_pred.shape[0]):
#         pre = torch.sum(torch.multiply(y_pred[i], y_true[i]))/(torch.sum(y_pred[i])+1E-9)
#         rec = torch.sum(torch.multiply(y_pred[i], y_true[i]))/(torch.sum(y_true[i])+1E-9)
#         F1.append(2 * pre * rec / (pre + rec+1E-9))

#     return mean(F1)

# def f1_score_calculation(y_pred, y_true):
#     y_pred = y_pred.reshape(1, -1)
#     y_true = y_true.reshape(1, -1)
#     pre = torch.sum(torch.multiply(y_pred, y_true))/(torch.sum(y_pred)+1E-9)
#     rec = torch.sum(torch.multiply(y_pred, y_true))/(torch.sum(y_true)+1E-9)
#     F1 = 2 * pre * rec / (pre + rec + 1E-9)
#     print("recall: ", rec, "pre: ", pre)
#     return F1


def structure_metrics_calculation(y_pred_binary, g_dgl: dgl.DGLGraph):
    """
    计算结构性评价指标（密度、直径、传导性）

    参数:
        y_pred_binary (Tensor): 预测的社区节点二值掩码，形状为 (query_num, node_num)
        g_dgl (dgl.DGLGraph): 图结构

    返回:
        avg_density (float)
        avg_diameter (float)
        avg_conductance (float)
    """
    # 转换为 igraph 图
    src, dst = g_dgl.edges()
    edges = list(zip(src.numpy(), dst.numpy()))
    n_nodes = g_dgl.num_nodes()
    ig_g = ig.Graph()
    ig_g.add_vertices(n_nodes)
    ig_g.add_edges(edges)
    ig_g = ig_g.as_undirected()

    y_pred = y_pred_binary.reshape(-1, y_pred_binary.shape[-1])

    density_list, diameter_list, conductance_list = [], [], []

    for i in range(y_pred.shape[0]):
        pred = y_pred[i].cpu().numpy().astype(bool)
        community_nodes = np.where(pred)[0]
        community_nodes_list = community_nodes.tolist()
        node_set = set(community_nodes_list)

        subgraph = ig_g.subgraph(community_nodes_list)
        n = len(community_nodes_list)
        m = subgraph.ecount()

        # 密度
        density = 0 if n <= 1 else 2 * m / (n * (n - 1))
        density_list.append(density)

        # 直径
        diameter = 0 if n == 0 or m == 0 else subgraph.diameter(directed=False, unconn=True)
        diameter_list.append(diameter)

        # 传导性
        boundary_edges = 0
        vol_in = 0
        vol_out = 0
        for v in range(ig_g.vcount()):
            deg = ig_g.degree(v)
            if v in node_set:
                vol_in += deg
                for nbr in ig_g.neighbors(v):
                    if nbr not in node_set:
                        boundary_edges += 1
            else:
                vol_out += deg
        conductance = boundary_edges / min(vol_in, vol_out) if min(vol_in, vol_out) > 0 else 0
        conductance_list.append(conductance)

    return np.mean(density_list), np.mean(diameter_list), np.mean(conductance_list)

def f1_score_calculation(y_pred_binary, y_true_binary):
    """
    计算社区搜索的四项指标（手动实现Jaccard）

    参数:
        y_pred_binary (Tensor): 预测的社区节点二值掩码，形状为 (query_num, node_num)
        y_true_binary (Tensor): 真实的社区节点二值掩码，形状同 y_pred_binary

    返回:
        avg_precision (float): 平均查准率
        avg_recall (float): 平均查全率
        avg_f1 (float): 平均F1分数
        avg_jaccard (float): 平均Jaccard相似度
    """
    # 确保输入为二维张量
    y_pred = y_pred_binary.reshape(-1, y_pred_binary.shape[-1])
    y_true = y_true_binary.reshape(-1, y_true_binary.shape[-1])

    precision_list = []
    recall_list = []
    f1_list = []
    jaccard_list = []

    for i in range(y_true.shape[0]):
        # 转换为numpy数组
        pred = y_pred[i].cpu().numpy().astype(bool)  # 转换为布尔数组
        true = y_true[i].cpu().numpy().astype(bool)

        # 计算关键量
        intersection = np.sum(pred & true)  # |V_q ∩ V_t|
        union = np.sum(pred | true)  # |V_q ∪ V_t|
        pred_size = np.sum(pred)  # |V_q|
        true_size = np.sum(true)  # |V_t|

        # 查准率 (公式3-26)
        precision = intersection / (pred_size + 1e-8)
        precision_list.append(precision)

        # 查全率 (公式3-27修正版)
        recall = intersection / (true_size + 1e-8)
        recall_list.append(recall)

        # F1分数
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_list.append(f1)

        # Jaccard相似度 (公式3-29)
        jaccard = intersection / (union + 1e-8)
        jaccard_list.append(jaccard)

    # 计算平均值
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_jaccard = np.mean(jaccard_list)

    return avg_precision, avg_recall, avg_f1, avg_jaccard

def evaluation(comm_find, comm):
    nmi_list, ari_list, jaccard_list = [], [], []
    for i in range(comm.shape[0]):  # 逐查询计算
        nmi = normalized_mutual_info_score(comm[i], comm_find[i])
        ari = adjusted_rand_score(comm[i], comm_find[i])
        jaccard = jaccard_score(comm[i], comm_find[i])
        nmi_list.append(nmi)
        ari_list.append(ari)
        jaccard_list.append(jaccard)
    return np.mean(nmi_list), np.mean(ari_list), np.mean(jaccard_list)

# def evaluation(comm_find, comm):
#
#     comm_find = comm_find.reshape(-1)
#     comm = comm.reshape(-1)
#
#     return normalized_mutual_info_score(comm, comm_find), adjusted_rand_score(comm, comm_find), jaccard_score(comm, comm_find)

# def evaluation(comm_find, comm):

#     nmi_all, ari_all, jac_all = [], [], []

#     for i in range(comm_find.shape[0]):
#         nmi_all.append(NMI_score(comm_find[i], comm[i]))
#         ari_all.append(ARI_score(comm_find[i], comm[i]))
#         jac_all.append(JAC_score(comm_find[i], comm[i]))

#     return np.mean(nmi_all), np.mean(ari_all), np.mean(jac_all) 

def NMI_score(comm_find, comm):

    score = normalized_mutual_info_score(comm, comm_find)
    #print("q, nmi:", score)
    return score

def ARI_score(comm_find, comm):

    score = adjusted_rand_score(comm, comm_find)
    #print("q, ari:", score)

    return score

def JAC_score(comm_find, comm):

    score = jaccard_score(comm, comm_find)
    #print("q, jac:", score)
    return score

def load_query_n_gt(path, dataset, vec_length):
    # load query and ground truth
    """
     加载查询向量（query）及其对应的真实结果（ground truth）

     参数：
         path (str): 数据集根目录路径（需确保以'/'结尾，如 'data/'）
         dataset (str): 数据集名称（对应根目录下的子目录名）
         vec_length (int): 输出向量的维度，需与实际数据中的最大索引+1匹配

     返回：
         query (torch.Tensor): 查询向量集合，形状为 [num_queries, vec_length]
                               dtype为torch.float32，元素为0或1的二进制向量
         gt (torch.Tensor): 真实结果向量集合，形状为 [num_queries, vec_length]
                            dtype为torch.float32，元素为0或1的二进制向量

     注意：
         1. 输入文件格式要求：
            - 文件路径结构：`{path}/{dataset}/{dataset}.query` 和 `{dataset}.gt`
            - 文件内容格式：每行为空格分隔的整数索引，例如 "3 5 7" 表示第3、5、7位置为1
         2. 需确保vec_length ≥ 数据中出现的最大索引+1，否则会引发索引越界错误
         3. 若文件不存在会触发FileNotFoundError
     """
    query = []
    # 打开查询文件（格式：每行包含空格分隔的索引，如 "0 3 5"）
    file_query = open(path + dataset + '/' + dataset + ".query", 'r')
    # 逐行处理查询数据
    for line in file_query:
        # 创建全零向量，长度由vec_length指定
        vec = [0 for i in range(vec_length)]
        # 去除首尾空白字符并按空格分割成索引列表
        line = line.strip()
        line = line.split(" ")
        # 将对应索引位置设为1
        for i in line:
            vec[int(i)] = 1
        query.append(vec)

    # 初始化真实结果向量列表（处理逻辑与query相同）
    gt = []
    file_gt = open(path + dataset + '/' + dataset + ".gt", 'r')     # 读取真实社区（每一行是当前查询节点对应的社区）
    for line in file_gt:
        vec = [0 for i in range(vec_length)]
        line = line.strip()
        line = line.split(" ")
        
        for i in line:
            vec[int(i)] = 1
        gt.append(vec)
    
    return torch.Tensor(query), torch.Tensor(gt)

def get_gt_legnth(path, dataset):
    gt_legnth = []
    file_gt = open(path + dataset + '/' + dataset + ".gt", 'r')     # 读取真实社区
    for line in file_gt:
        line = line.strip()
        line = line.split(" ")
        gt_legnth.append(len(line))
    
    return torch.Tensor(gt_legnth)

def cosin_similarity(query_tensor, emb_tensor):
    # similarity = torch.stack([torch.cosine_similarity(query_tensor[i], emb_tensor, dim=1) for i in range(len(query_tensor))], 0)
    similarity = torch.stack([torch.cosine_similarity(query_tensor[i].reshape(1, -1), emb_tensor, dim=1) for i in range(len(query_tensor))], 0)
    # print(similarity.shape)
    return similarity
    
def dot_similarity(query_tensor, emb_tensor):
    similarity = torch.mm(query_tensor, emb_tensor.t()) # (query_num, node_num)
    similarity = torch.nn.Softmax(dim=1)(similarity)
    return similarity


def transform_coo_to_csr(adj):
    """
    将 PyTorch 的 COO 格式稀疏张量 (torch.sparse.LongTensor) 转换为 SciPy 的 CSR 格式稀疏矩阵 (scipy.sparse.csr_matrix)。

    输入:
    - adj (torch.sparse.LongTensor): 输入的稀疏张量，格式为 PyTorch 的 COO 格式。
        - adj._indices() 包含非零值的行和列索引。
        - adj._values() 包含非零值的具体值。
        - adj.size() 表示稀疏张量的形状。

    输出:
    - adj (scipy.sparse.csr_matrix): 转换后的稀疏矩阵，格式为 SciPy 的 CSR (Compressed Sparse Row) 格式。

    功能:
    - 从 PyTorch 的 COO 稀疏张量中提取数据 (行索引、列索引、非零值)。
    - 使用 SciPy 的 `csr_matrix` 构造一个新的稀疏矩阵。
    """

    # 提取 COO 稀疏张量中的行索引和列索引
    row = adj._indices()[0]  # 行索引 (Tensor)
    col = adj._indices()[1]  # 列索引 (Tensor)

    # 提取非零值
    data = adj._values()  # 非零值 (Tensor)

    # 提取稀疏张量的形状
    shape = adj.size()  # 稀疏张量的整体形状 (size: tuple)

    # 构造 SciPy 的 CSR 格式稀疏矩阵
    adj = sp.csr_matrix((data, (row, col)), shape=shape)
    # 参数:
    # - data: 非零值的内容。
    # - (row, col): 非零值的位置 (行和列索引)。
    # - shape: 矩阵的形状。

    # 返回构造的 CSR 矩阵
    return adj

def transform_csr_to_coo(adj, size=None):
    """
    将一个稀疏矩阵 (scipy.sparse.csr_matrix) 转换为 PyTorch 支持的 COO 格式稀疏张量 (torch.sparse.LongTensor)。

    输入:
    - adj (scipy.sparse.csr_matrix): 输入的稀疏矩阵，格式为 CSR (Compressed Sparse Row)。
    - size (int, 可选): 稀疏矩阵的大小，用于设置输出张量的形状。如果为 None，则使用输入矩阵的默认大小。

    输出:
    - adj (torch.sparse.LongTensor): 转换后的稀疏张量，格式为 PyTorch COO 格式。

    功能:
    - 将 SciPy 的 CSR 格式稀疏矩阵转换为 PyTorch 的 COO 格式稀疏张量，供后续计算使用。
    """

    # 将 CSR 格式的矩阵转换为 COO 格式
    adj = adj.tocoo()
    # 在 COO 格式中，矩阵存储为 (row, col, data)，分别表示非零元素的行索引、列索引和值。

    # 创建 PyTorch 稀疏张量
    adj = torch.sparse.LongTensor(
        torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),  # 转换行和列索引为 PyTorch LongTensor
        torch.LongTensor(adj.data.astype(np.int32)),            # 转换数据为 PyTorch LongTensor
        torch.Size([size, size])                                # 指定稀疏张量的大小 (size x size)
    )

    # 返回转换后的 PyTorch 稀疏张量
    return adj

def transform_sp_csr_to_coo(adj, batch_size, node_num):
    """
    将一个稀疏邻接矩阵 (sp_csr 格式) 按批次大小分块，并将分块结果转换为 COO 格式的稠密张量。

    输入:
    - adj (scipy.sparse.csr_matrix): 邻接矩阵，形状为 (node_num, node_num)。表示图的稀疏邻接矩阵。
    - batch_size (int): 每一批次的节点数。
    - node_num (int): 图中总的节点数。

    输出:
    - adj_tensor_coo (list of torch.Tensor): 按块处理后的邻接矩阵，稠密张量格式，形状为 batch_num个(batch_size, batch_size)。 最后一个可能不是(batch_size, batch_size)
    - minus_adj_tensor_coo (list of torch.Tensor): 按块处理后的“反邻接矩阵”，稠密张量格式，形状为 batch_num个(batch_size, batch_size)。最后一个可能不是(batch_size, batch_size)

    功能:
    - 将输入的稀疏邻接矩阵分块，每一块只包含当前批次节点之间的子图信息。
    - 转换每一块为 COO 格式的稠密张量，并计算“反邻接矩阵”。
    """
    # chunks    划分节点索引为块
    node_index = [i for i in range(node_num)]
    divide_index = [node_index[i:i+batch_size] for i in range(0, len(node_index), batch_size)]

    # 提取每个块的子矩阵 (邻接矩阵)，格式为 sp_csr
    print("开始小批处理：提取每块的邻接矩阵")
    adj_sp_csr = [adj[divide_index[i]][:, divide_index[i]] for i in range(len(divide_index))]   # 从原始邻接矩阵中提取当前批次块的子矩阵，仅包含当前批次内节点之间的边。
    # adj_sp_csr 是一个长度为 batch_num 的列表。列表中每个元素是一个稀疏矩阵，形状为 (batch_size_i, batch_size_i)，其中： batch_size_i = batch_size（对于完整块）。 batch_size_i <= batch_size（对于最后一块）。

    # 计算“反邻接矩阵” (1-邻接矩阵)
    print("开始小批处理：计算每块的反邻接矩阵")
    minus_adj_sp_csr = [sp.csr_matrix(torch.ones(item.shape))-item for item in adj_sp_csr]      # 使用全 1 矩阵减去当前块的邻接矩阵，得到“反邻接矩阵”。

    # adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in adj_sp_csr]
    # minus_adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in minus_adj_sp_csr]
    # 转换邻接矩阵到 COO 格式的稠密张量
    print("开始小批处理：将邻接矩阵转换为稠密 COO 格式")
    adj_tensor_coo = [transform_csr_to_coo(adj_sp_csr[i], len(divide_index[i])).to_dense() for i in range(len(divide_index))]
    # 将每块 CSR 格式的矩阵转换为 COO 格式，并转为稠密张量，形状为 (batch_size, batch_size)。

    # 转换“反邻接矩阵”到 COO 格式的稠密张量
    print("开始小批处理：将反邻接矩阵转换为稠密 COO 格式")
    minus_adj_tensor_coo = [transform_csr_to_coo(minus_adj_sp_csr[i], len(divide_index[i])).to_dense() for i in range(len(divide_index))]
    # 将每块“反邻接矩阵”从 CSR 格式转换为 COO 格式，并转为稠密张量。

    return adj_tensor_coo, minus_adj_tensor_coo


# transform coo to edge index in pytorch geometric 
def transform_coo_to_edge_index(adj):
    adj = adj.coalesce()
    edge_index = adj.indices().detach().long()
    return edge_index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_adj_to_scipy(adj):

    shape = adj.shape
    coords = adj.coalesce().indices()
    values = adj.coalesce().values()

    scipy_sparse = sp.coo_matrix((values.cpu().numpy(), (coords[0].cpu().numpy(), coords[1].cpu().numpy())), shape=shape)

    return scipy_sparse

# determine one edge in edge_index or not of torch geometric
def is_edge_in_edge_index(edge_index, source, target):
    mask = (edge_index[0] == source) & (edge_index[1] == target)
    return mask.any()

def construct_pseudo_assignment(cluster_ids_x):
    pseudo_assignment = torch.zeros(cluster_ids_x.shape[0], int(cluster_ids_x.max()+1))

    for i in range(cluster_ids_x.shape[0]):
        pseudo_assignment[i][int(cluster_ids_x[i])] = 1
    
    return pseudo_assignment

def pq_computation(similarity):
    q = torch.nn.functional.normalize(similarity, dim=1, p=1)
    p_temp = torch.mul(q, q)
    q_colsum = torch.sum(q, axis=0)
    p_temp = torch.div(p_temp,q_colsum)
    p = torch.nn.functional.normalize(p_temp, dim=1, p=1)
    return q, p

def coo_matrix_to_nx_graph(matrix):
    # Create an empty NetworkX graph
    graph = nx.Graph()

    # Get the number of nodes in the COO matrix
    num_nodes = matrix.shape[0]

    # Convert the COO matrix to a dense matrix
    dense_matrix = matrix.to_dense()

    # Iterate over the non-zero entries in the dense matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if dense_matrix[i][j] != 0:
                # Add an edge to the NetworkX graph
                graph.add_edge(i, j)
                graph.add_edge(j, i)

    return graph

def coo_matrix_to_nx_graph_efficient(adj_matrix):
    # 创建一个无向图对象
    graph = nx.Graph()

    # 获取 COO 矩阵的行和列索引以及权重值
    adj_matrix = adj_matrix.coalesce()
    rows = adj_matrix.indices()[0]
    cols = adj_matrix.indices()[1]

    # 添加节点和边到图中
    for i in range(len(rows)):
        graph.add_edge(int(rows[i]), int(cols[i]))
        graph.add_edge(int(cols[i]), int(rows[i]))

    return graph

def obtain_adj_from_nx(graph):
    return np.array(nx.adjacency_matrix(graph, nodelist=[i for i in range(max(graph.nodes)+1)]).todense())

def find_all_neighbors_bynx(query, Graph):
    
    nodes = Graph.nodes()

    neighbors = []
    for i in range(len(query)):
        if query[i] not in nodes:
            continue
        for j in Graph.neighbors(query[i]):
            if j not in query:
                neighbors.append(j)
    return neighbors

def MaxMinNormalization(x, Min, Max):
    
    x = np.array(x)
    x_max = np.max(x)
    x_min = np.min(x)

    x = [(item-x_min)*(Max-Min)/(x_max - x_min) + Min for item in x]

    return x

