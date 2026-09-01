import utils  # 导入自定义工具函数模块
import dgl  # 导入 DGL (Deep Graph Library) 用于图数据处理
import torch  # 导入 PyTorch 库，用于深度学习
import scipy.sparse as sp  # 导入 scipy 库的稀疏矩阵模块

# 导入DGL数据集模块，包含多个公开的图数据集
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset

import numpy as np
import os.path as osp

from sklearn.preprocessing import OneHotEncoder
import pickle
from typing import List, Tuple, Dict, Union
import json


# def get_dataset1(dataset_name, pe_dim):
#     """
#     修改后的兼容版本，确保输出与原始函数一致
#     """
#
#     # --------------------------
#     # 文件路径处理（保持新函数逻辑）
#     # --------------------------
#     data_dir = osp.join("dataset", dataset_name)
#     adj_path = osp.join(data_dir, "final_meta_path.npz")
#     feat_path = osp.join(data_dir, "feat.npz")
#
#     # --------------------------
#     # 邻接矩阵处理（兼容性关键修改）
#     # --------------------------
#     # 加载稀疏邻接矩阵并转换为稠密LongTensor
#     adj_sparse = sp.load_npz(adj_path)
#     adj_dense = torch.tensor(
#         adj_sparse.todense(),
#         dtype=torch.long  # 强制转换为LongTensor以兼容原函数
#     )
#
#     # 构建DGL图（用于后续LPE计算）
#     src_nodes = torch.tensor(adj_sparse.row, dtype=torch.long)
#     dst_nodes = torch.tensor(adj_sparse.col, dtype=torch.long)
#     graph = dgl.graph((src_nodes, dst_nodes))
#     graph = dgl.to_bidirected(graph)  # 确保无向图
#
#
#     # --------------------------
#     # 特征处理（修复数据类型问题）
#     # --------------------------
#     # 加载特征并保持float32类型
#     feat_sparse = sp.load_npz(feat_path)
#     features = torch.FloatTensor(feat_sparse.todense()) if sp.issparse(feat_sparse) \
#         else torch.FloatTensor(feat_sparse)
#
#     # 计算LPE并拼接
#     lpe = utils.laplacian_positional_encoding(graph, pe_dim)
#     combined_features = torch.cat([features, lpe], dim=1)
#
#     # --------------------------
#     # 最终输出对齐原始函数
#     # --------------------------
#     return adj_dense.cpu().type(torch.LongTensor), combined_features.long()           # 保持与原始函数一致的类型转换

def get_dataset1(dataset_name, pe_dim):
    data_dir = osp.join("dataset", dataset_name)

    if dataset_name == 'acm2':
        adj_path = osp.join(data_dir, "adj.npz")
        # # --------------------------
        # # 1. 邻接矩阵处理（去自环）
        # # --------------------------
        # adj_sparse = sp.load_npz(adj_path)
        # graph = dgl.from_scipy(adj_sparse)
        # graph = dgl.remove_self_loop(graph)  # 去除自环
        # graph = dgl.to_bidirected(graph)  # 确保无向图
        #
        # # 转换为PyTorch稀疏张量
        # adj_coo = graph.adjacency_matrix()
        # indices = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
        # values = torch.ones(adj_coo.nnz, dtype=torch.float)
        # adj = torch.sparse_coo_tensor(indices, values, adj_coo.shape)

        adj_sparse = sp.load_npz(adj_path).tocoo()
        # 1. 过滤自环边（行索引 != 列索引）
        mask = (adj_sparse.row != adj_sparse.col)
        filtered_rows = adj_sparse.row[mask]
        filtered_cols = adj_sparse.col[mask]

        # 2. 强制生成双向边（确保每个边i-j都有j-i）
        # 将原始边和反向边合并（例如：原始边是0-1 → 添加1-0）
        edges = np.stack([filtered_rows, filtered_cols], axis=0)  # 原始边 (2, num_edges)
        reverse_edges = edges[[1, 0], :]  # 反向边 (2, num_edges)
        bidir_edges = np.concatenate([edges, reverse_edges], axis=1)  # 合并 (2, 2*num_edges)

        # 3. 去重（避免原始邻接矩阵已经包含双向边时重复）
        bidir_edges_unique = np.unique(bidir_edges, axis=1)  # 自动去重
        bidir_rows = bidir_edges_unique[0, :]
        bidir_cols = bidir_edges_unique[1, :]

        # 转换为COO格式的PyTorch稀疏张量（双向且无自环）
        rows = torch.tensor(bidir_rows, dtype=torch.long)
        cols = torch.tensor(bidir_cols, dtype=torch.long)
        indices = torch.stack([rows, cols], dim=0)
        values = torch.ones_like(rows, dtype=torch.float)
        adj = torch.sparse_coo_tensor(indices, values, adj_sparse.shape)  # shape保持不变

        # 构建DGL图（双向且无自环）
        graph = dgl.graph((rows, cols))  # 不需要再调用to_bidirected，已显式包含双向边


        # --------------------------
        # 2. 特征处理（自动对齐维度）
        # --------------------------
        # 加载各类型特征（假设顺序：Paper -> Author -> Subject）
        p_feat = torch.FloatTensor(sp.load_npz(osp.join(data_dir, 'p_feat.npz')).todense())  # Paper
        a_feat = torch.FloatTensor(sp.load_npz(osp.join(data_dir, 'a_feat.npz')).todense())  # Author
        s_feat = torch.eye(graph.num_nodes() - p_feat.shape[0] - a_feat.shape[0])  # Subject

        # 对齐特征维度（补零）
        max_dim = max(p_feat.shape[1], a_feat.shape[1], s_feat.shape[1])

        def pad_features(feat):
            return torch.cat([feat, torch.zeros(feat.shape[0], max_dim - feat.shape[1])], dim=1)

        # 按顺序合并特征矩阵
        total_features = torch.cat([
            pad_features(p_feat),
            pad_features(a_feat),
            pad_features(s_feat)
        ], dim=0)

        # --------------------------
        # 3. 拼接位置编码
        # --------------------------
        lpe = utils.laplacian_positional_encoding(graph, pe_dim)
        combined_features = torch.cat([total_features, lpe], dim=1)

        return adj.coalesce().cpu(), combined_features
    elif dataset_name == 'dblp2':
        # --------------------------
        # 1. 邻接矩阵处理（去自环）
        # --------------------------
        adj_path = osp.join(data_dir, "adj.npz")
        # adj_sparse = sp.load_npz(adj_path)
        # graph = dgl.from_scipy(adj_sparse)
        # graph = dgl.remove_self_loop(graph)
        # graph = dgl.to_bidirected(graph)
        #
        # # 转换为PyTorch稀疏张量
        # adj_coo = graph.adjacency_matrix()
        # indices = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
        # values = torch.ones(adj_coo.nnz, dtype=torch.float)
        # adj = torch.sparse_coo_tensor(indices, values, adj_coo.shape)
        # 加载稀疏邻接矩阵并转换为COO格式
        adj_sparse = sp.load_npz(adj_path).tocoo()

        # 1. 过滤自环边（行索引 != 列索引）
        mask = (adj_sparse.row != adj_sparse.col)
        filtered_rows = adj_sparse.row[mask]
        filtered_cols = adj_sparse.col[mask]

        # 2. 强制生成双向边（确保每个边i-j都有j-i）
        # 将原始边和反向边合并（例如：原始边是0-1 → 添加1-0）
        edges = np.stack([filtered_rows, filtered_cols], axis=0)  # 原始边 (2, num_edges)
        reverse_edges = edges[[1, 0], :]  # 反向边 (2, num_edges)
        bidir_edges = np.concatenate([edges, reverse_edges], axis=1)  # 合并 (2, 2*num_edges)

        # 3. 去重（避免原始邻接矩阵已经包含双向边时重复）
        bidir_edges_unique = np.unique(bidir_edges, axis=1)  # 自动去重
        bidir_rows = bidir_edges_unique[0, :]
        bidir_cols = bidir_edges_unique[1, :]

        # 转换为COO格式的PyTorch稀疏张量（双向且无自环）
        rows = torch.tensor(bidir_rows, dtype=torch.long)
        cols = torch.tensor(bidir_cols, dtype=torch.long)
        indices = torch.stack([rows, cols], dim=0)
        values = torch.ones_like(rows, dtype=torch.float)
        adj = torch.sparse_coo_tensor(indices, values, adj_sparse.shape)  # shape保持不变

        # 构建DGL图（双向且无自环）
        graph = dgl.graph((rows, cols))  # 不需要再调用to_bidirected，已显式包含双向边




        # --------------------------
        # 2. 特征处理（四类节点对齐）
        # --------------------------
        # 加载已知特征文件
        a_feat = torch.FloatTensor(sp.load_npz(osp.join(data_dir, 'a_feat.npz')).todense())  # 作者
        p_feat = torch.FloatTensor(sp.load_npz(osp.join(data_dir, 'p_feat.npz')).todense())  # 论文
        t_feat = torch.FloatTensor(np.load(osp.join(data_dir, 't_feat.npz')))  # 术语

        # 推导会议节点数量并生成单位矩阵特征
        num_c = graph.num_nodes() - a_feat.shape[0] - p_feat.shape[0] - t_feat.shape[0]
        c_feat = torch.eye(num_c, dtype=torch.float32)  # 会议特征

        # 特征维度对齐（补零）
        max_dim = max(a_feat.shape[1], p_feat.shape[1], t_feat.shape[1], c_feat.shape[1])

        def pad_features(feat):
            return torch.cat([feat, torch.zeros(feat.shape[0], max_dim - feat.shape[1])], dim=1)

        # 按节点顺序合并特征：作者 -> 论文 -> 术语 -> 会议
        total_features = torch.cat([
            pad_features(a_feat),
            pad_features(p_feat),
            pad_features(t_feat),
            pad_features(c_feat)
        ], dim=0)

        # --------------------------
        # 3. 拼接位置编码
        # --------------------------
        lpe = utils.laplacian_positional_encoding(graph, pe_dim)
        combined_features = torch.cat([total_features, lpe], dim=1)

        return adj.coalesce().cpu(), combined_features

    elif dataset_name == 'imdb2':  # 新增self_IMDB分支
        # --------------------------
        # 1. 邻接矩阵处理（去自环）
        # --------------------------
        adj_path = osp.join(data_dir, "adj.npz")
        # adj_sparse = sp.load_npz(adj_path)
        # graph = dgl.from_scipy(adj_sparse)
        # graph = dgl.remove_self_loop(graph)
        # graph = dgl.to_bidirected(graph)
        #
        # # 转换为PyTorch稀疏张量
        # adj_coo = graph.adjacency_matrix()
        # indices = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
        # values = torch.ones(adj_coo.nnz, dtype=torch.float)
        # adj = torch.sparse_coo_tensor(indices, values, adj_coo.shape)

        # 加载稀疏邻接矩阵并转换为COO格式
        adj_sparse = sp.load_npz(adj_path).tocoo()

        # 1. 过滤自环边（行索引 != 列索引）
        mask = (adj_sparse.row != adj_sparse.col)
        filtered_rows = adj_sparse.row[mask]
        filtered_cols = adj_sparse.col[mask]

        # 2. 强制生成双向边（确保每个边i-j都有j-i）
        # 将原始边和反向边合并（例如：原始边是0-1 → 添加1-0）
        edges = np.stack([filtered_rows, filtered_cols], axis=0)  # 原始边 (2, num_edges)
        reverse_edges = edges[[1, 0], :]  # 反向边 (2, num_edges)
        bidir_edges = np.concatenate([edges, reverse_edges], axis=1)  # 合并 (2, 2*num_edges)

        # 3. 去重（避免原始邻接矩阵已经包含双向边时重复）
        bidir_edges_unique = np.unique(bidir_edges, axis=1)  # 自动去重
        bidir_rows = bidir_edges_unique[0, :]
        bidir_cols = bidir_edges_unique[1, :]

        # 转换为COO格式的PyTorch稀疏张量（双向且无自环）
        rows = torch.tensor(bidir_rows, dtype=torch.long)
        cols = torch.tensor(bidir_cols, dtype=torch.long)
        indices = torch.stack([rows, cols], dim=0)
        values = torch.ones_like(rows, dtype=torch.float)
        adj = torch.sparse_coo_tensor(indices, values, adj_sparse.shape)  # shape保持不变

        # 构建DGL图（双向且无自环）
        graph = dgl.graph((rows, cols))  # 不需要再调用to_bidirected，已显式包含双向边






        # --------------------------
        # 2. 特征处理（三类节点对齐）
        # --------------------------
        # 加载已知特征文件
        m_feat = torch.FloatTensor(sp.load_npz(osp.join(data_dir, 'm_feat.npz')).todense())  # 电影

        # 推导演员和导演节点数量（假设顺序：电影->演员->导演）
        remaining_nodes = graph.num_nodes() - m_feat.shape[0]
        remaining_feat = torch.eye(remaining_nodes, dtype=torch.float32)  # 演员特征（单位矩阵）

        # 特征维度对齐（补零）
        max_dim = max(m_feat.shape[1], remaining_feat.shape[1])

        def pad_features(feat):
            return torch.cat([feat, torch.zeros(feat.shape[0], max_dim - feat.shape[1])], dim=1)

        # 按顺序合并特征矩阵：电影 -> 演员 -> 导演
        total_features = torch.cat([
            pad_features(m_feat),
            pad_features(remaining_feat),
        ], dim=0)

        # --------------------------
        # 3. 拼接位置编码
        # --------------------------
        lpe = utils.laplacian_positional_encoding(graph, pe_dim)
        combined_features = torch.cat([total_features, lpe], dim=1)

        return adj.coalesce().cpu(), combined_features

    else:
        adj_path = osp.join(data_dir, "final_meta_path.npz")
        feat_path = osp.join(data_dir, "feat.npz")

        # # --------------------------
        # # 邻接矩阵处理（关键修改）
        # # --------------------------
        # # 加载稀疏邻接矩阵并转换为PyTorch稀疏张量
        # adj_sparse = sp.load_npz(adj_path).tocoo()
        #
        # # 转换为COO格式的PyTorch稀疏张量
        # rows = torch.tensor(adj_sparse.row, dtype=torch.long)
        # cols = torch.tensor(adj_sparse.col, dtype=torch.long)
        # indices = torch.stack([rows, cols], dim=0)
        # values = torch.ones_like(rows, dtype=torch.float)  # 假设边权重为1
        # adj = torch.sparse_coo_tensor(indices, values, adj_sparse.shape)
        #
        # # 构建DGL图
        # graph = dgl.graph((rows, cols))
        # graph = dgl.to_bidirected(graph)

        # --------------------------
        # 邻接矩阵处理（关键修改）
        # --------------------------
        # 加载稀疏邻接矩阵并转换为COO格式
        adj_sparse = sp.load_npz(adj_path).tocoo()

        # 1. 过滤自环边（行索引 != 列索引）
        mask = (adj_sparse.row != adj_sparse.col)
        filtered_rows = adj_sparse.row[mask]
        filtered_cols = adj_sparse.col[mask]

        # 2. 强制生成双向边（确保每个边i-j都有j-i）
        # 将原始边和反向边合并（例如：原始边是0-1 → 添加1-0）
        edges = np.stack([filtered_rows, filtered_cols], axis=0)  # 原始边 (2, num_edges)
        reverse_edges = edges[[1, 0], :]  # 反向边 (2, num_edges)
        bidir_edges = np.concatenate([edges, reverse_edges], axis=1)  # 合并 (2, 2*num_edges)

        # 3. 去重（避免原始邻接矩阵已经包含双向边时重复）
        bidir_edges_unique = np.unique(bidir_edges, axis=1)  # 自动去重
        bidir_rows = bidir_edges_unique[0, :]
        bidir_cols = bidir_edges_unique[1, :]

        # 转换为COO格式的PyTorch稀疏张量（双向且无自环）
        rows = torch.tensor(bidir_rows, dtype=torch.long)
        cols = torch.tensor(bidir_cols, dtype=torch.long)
        indices = torch.stack([rows, cols], dim=0)
        values = torch.ones_like(rows, dtype=torch.float)
        adj = torch.sparse_coo_tensor(indices, values, adj_sparse.shape)  # shape保持不变

        # 构建DGL图（双向且无自环）
        graph = dgl.graph((rows, cols))  # 不需要再调用to_bidirected，已显式包含双向边

        # --------------------------
        # 特征处理
        # --------------------------
        # 加载特征并保持float32类型
        feat_sparse = sp.load_npz(feat_path)
        features = torch.FloatTensor(feat_sparse.todense()) if sp.issparse(feat_sparse) \
            else torch.FloatTensor(feat_sparse)

        # 计算LPE并拼接
        lpe = utils.laplacian_positional_encoding(graph, pe_dim)
        combined_features = torch.cat([features, lpe], dim=1)

        # --------------------------
        # 最终输出
        # --------------------------
        # 保持稀疏张量和特征的数据类型
        return adj.coalesce().cpu(), combined_features  # 移除.long()转换，保持特征为float




def get_dataset(dataset, pe_dim):
    """
    获取指定数据集及其特征和邻接矩阵。

    参数：
        dataset (str): 数据集名称，如 "pubmed", "photo", "cs", "cora", "physics", "citeseer" 等
        pe_dim (int): 用于计算图的拉普拉斯位置编码（Laplacian Positional Encoding）的维度

    返回：
        adj (torch.Tensor): 图的邻接矩阵，形状为 [num_nodes, num_nodes]
        features (torch.Tensor): 图节点的特征矩阵，形状为 [num_nodes, num_features]，其中 num_features = 原特征维度 + pe_dim
    """

    # 如果数据集是 DGL 数据集（如 pubmed, photo, cs 等）
    if dataset in {"pubmed", "photo", "cs", "cora", "physics", "citeseer"}:

        # 根据数据集名称选择存储数据的文件路径
        if dataset in {"photo", "cs"}:
            file_path = "dataset/" + dataset + "_dgl.pt"  # 对于 photo 和 cs 数据集，使用 dgl 格式存储
        else:
            file_path = "dataset/" + dataset + "_pyg.pt"  # 对于其他数据集，使用 pyg 格式存储

        # 加载数据文件
        data_list = torch.load(file_path)

        # 获取邻接矩阵和特征矩阵
        adj = data_list[0]  # 邻接矩阵的形状为 [num_nodes, num_nodes]
        features = data_list[1]  # 特征矩阵的形状为 [num_nodes, num_features]

        # 根据数据集名称加载图结构
        if dataset == "pubmed":
            graph = PubmedGraphDataset()[0]
        elif dataset == "photo":
            graph = AmazonCoBuyPhotoDataset()[0]
        elif dataset == "cs":
            graph = CoauthorCSDataset()[0]
        elif dataset == "physics":
            graph = CoauthorPhysicsDataset()[0]
        elif dataset == "cora":
            graph = CoraGraphDataset()[0]
        elif dataset == "citeseer":
            graph = CiteseerGraphDataset()[0]

        # 转换图为双向图（每条边会有两个方向）
        graph = dgl.to_bidirected(graph)

        # 计算拉普拉斯位置编码（Laplacian Positional Encoding）
        lpe = utils.laplacian_positional_encoding(graph, pe_dim)  # lpe 形状为 [num_nodes, pe_dim]

        # 将位置编码与原始特征拼接在一起
        features = torch.cat((features, lpe), dim=1)  # 新的 features 形状为 [num_nodes, num_features + pe_dim]


    # 如果数据集是其他图数据集（如 texas, cornell, wisconsin, dblp, reddit 等）
    elif dataset in {"texas", "cornell", "wisconsin", "dblp", "reddit"}:
        # 加载数据文件
        file_path = "dataset/" + dataset + "_pyg.pt"
        data_list = torch.load(file_path)

        # 获取邻接矩阵和特征矩阵
        adj = data_list[0]  # 邻接矩阵的形状为 [num_nodes, num_nodes]
        features = data_list[1]  # 特征矩阵的形状为 [num_nodes, num_features]

        # 将邻接矩阵转换为 scipy 稀疏矩阵格式
        adj_scipy = utils.torch_adj_to_scipy(adj)

        # 将 scipy 稀疏矩阵转换为 DGL 图
        graph = dgl.from_scipy(adj_scipy)

        # 计算拉普拉斯位置编码
        lpe = utils.laplacian_positional_encoding(graph, pe_dim)  # lpe 形状为 [num_nodes, pe_dim]

        # 将位置编码与原始特征拼接在一起
        features = torch.cat((features, lpe), dim=1)  # 新的 features 形状为 [num_nodes, num_features + pe_dim]

    # 打印邻接矩阵和特征矩阵的类型，确保数据加载正确
    print(type(adj), type(features))

    # 返回邻接矩阵和特征矩阵，转换为 LongTensor 类型并移动到 CPU
    return adj.cpu().type(torch.LongTensor), features.long()
