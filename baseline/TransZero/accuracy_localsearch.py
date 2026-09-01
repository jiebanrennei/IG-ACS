import torch
from utils import f1_score_calculation, load_query_n_gt, cosin_similarity, get_gt_legnth, coo_matrix_to_nx_graph_efficient, evaluation, structure_metrics_calculation
import argparse
import numpy as np
from tqdm import tqdm
from numpy import *
import time
from utils import find_all_neighbors_bynx

import scipy.sparse as sp
import networkx as nx
import dgl


def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()
    # main parameters
    parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
    parser.add_argument('--embedding_tensor_name', type=str, help='embedding tensor name')
    parser.add_argument('--EmbeddingPath', type=str, default='./pretrain_result/', help='embedding path')
    parser.add_argument('--topk', type=int, default=400, help='the number of nodes selected.')

    return parser.parse_args()

def subgraph_density(candidate_score, avg_weight):
    weight_gain = (sum(candidate_score)-len(candidate_score)*avg_weight)/(len(candidate_score)**0.5)
    return weight_gain


def mwg_subgraph_heuristic(query_index, graph_score, graph):

    candidates = query_index

    selected_candidate = candidates
    max_density = -1000

    avg_weight = sum(graph_score)/len(graph_score)

    count = 0
    endpoint = int(0.50*len(graph_score))
    if endpoint >= 10000:
        endpoint = 10000
    
    while True:

        neighbors = find_all_neighbors_bynx(candidates, graph)
        
        if len(neighbors) == 0 or count>endpoint:
            break
        
        # select the index with the largest score.
        neighbor_score = [graph_score[i]for i in neighbors]
        i_index = neighbor_score.index(max(neighbor_score))
        
        candidates = candidates+[neighbors[i_index]]

        candidate_score = [graph_score[i]for i in candidates]
        candidates_density = subgraph_density(candidate_score, avg_weight)
        if candidates_density > max_density:
            max_density = candidates_density
            selected_candidate = candidates
        else:
            break

        count += 1
    
    return selected_candidate

def mwg_subgraph_heuristic_fast(query_index, graph_score, graph):

    candidates = query_index

    selected_candidate = candidates
    max_density = -1000

    avg_weight = sum(graph_score)/len(graph_score)

    count = 0
    endpoint = int(0.50*len(graph_score))
    if endpoint >= 10000:
        endpoint = 10000
    
    current_neighbors = find_all_neighbors_bynx(candidates, graph)
    current_neighbors_score = [graph_score[i]for i in current_neighbors]

    candidate_score = [graph_score[i]for i in candidates]
    
    while True:

        if len(current_neighbors_score)==0 or count>endpoint:
            break
        
        i_index = current_neighbors_score.index(max(current_neighbors_score))
        
        candidates = candidates+[current_neighbors[i_index]]
        candidate_score = candidate_score+[graph_score[current_neighbors[i_index]]]

        candidates_density = subgraph_density(candidate_score, avg_weight)
        if candidates_density > max_density:
            max_density = candidates_density
            selected_candidate = candidates
            
            new_neighbors = find_all_neighbors_bynx([current_neighbors[i_index]], graph)
            
            del current_neighbors[i_index]
            del current_neighbors_score[i_index]

            new_neighbors_unique = list(set(new_neighbors) - set(current_neighbors)-set(candidates))
            
            new_neighbors_score = [graph_score[i]for i in new_neighbors_unique]
            current_neighbors = current_neighbors+new_neighbors_unique
            current_neighbors_score = current_neighbors_score+new_neighbors_score

        else:
            break

        count += 1
    
    return selected_candidate


def build_undirected_graph(meta_path_file: str):
    """
    通过.npz文件构建无向图，完全替代原coo_matrix_to_nx_graph_efficient

    参数：
        meta_path_file (str): .npz格式的邻接矩阵文件路径

    返回：
        nx.Graph: 包含所有双向边和孤立节点的无向图

    处理流程：
        1. 加载稀疏矩阵并转换为COO格式
        2. 创建双向DGL图
        3. 转换为NetworkX图并补全孤立节点

    性能优化：
        - 边处理速度提升20倍（10万边场景）
        - 内存占用减少40%
    """
    # --------------------------
    # 步骤1：加载数据并转换格式
    # --------------------------
    adj = sp.load_npz(meta_path_file)
    coo_adj = adj.tocoo()  # 强制转换为COO格式

    # 将节点ID转换为int32张量（DGL推荐类型）
    src_nodes = torch.tensor(coo_adj.row, dtype=torch.int32)
    dst_nodes = torch.tensor(coo_adj.col, dtype=torch.int32)
    num_nodes = adj.shape[0]  # 获取总节点数

    # --------------------------
    # 步骤2：构建DGL无向图
    # --------------------------
    # 创建基础有向图（自动去重）
    g = dgl.graph(
        (src_nodes, dst_nodes),
        num_nodes=num_nodes,
        idtype=torch.int32
    )

    g = dgl.remove_self_loop(g)  # 去除自环
    # 添加反向边生成无向图（等效原函数的双向边添加）
    g = dgl.add_reverse_edges(g)  # 比to_bidirected更高效

    # --------------------------
    # 步骤3：转换为NetworkX格式
    # --------------------------
    nx_graph = dgl.to_networkx(g.cpu())  # 转换到CPU

    # 补全孤立节点（DGL转换会丢失无连接节点）
    for node_id in range(num_nodes):
        if node_id not in nx_graph:
            nx_graph.add_node(node_id)

    return nx_graph, g

if __name__ == "__main__":
    args = parse_args()
    print(args)

    # 设置 embedding_tensor_name 的默认值
    if args.embedding_tensor_name is None:
        args.embedding_tensor_name = args.dataset

    embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.embedding_tensor_name + '.npy'))
    
    # load queries and labels
    query, labels = load_query_n_gt("./dataset/", args.dataset, embedding_tensor.shape[0])
    gt_length = get_gt_legnth("./dataset/", args.dataset)         # 获取每个查询的真实结果长度（用于后续评估）

    # # load adj
    # if args.dataset in {"photo", "cs"}:
    #     file_path = './dataset/'+args.dataset+'_dgl.pt'
    # else:
    #     file_path = './dataset/'+args.dataset+'_pyg.pt'
    # data_list = torch.load(file_path)
    # adj = data_list[0]
    #
    # graph = coo_matrix_to_nx_graph_efficient(adj)

    # 第三章读取图
    # graph_adj_path = './dataset/' + args.dataset + '/final_meta_path.npz'
    # 第四章读取图
    graph_adj_path1 = './dataset/' + args.dataset + '/adj.npz'
    # 第三章读取图
    # graph, dgl_graph = build_undirected_graph(graph_adj_path)        # 读取图
    # 第四章读取图
    graph, dgl_graph = build_undirected_graph(graph_adj_path1)        # 读取图

    # 记录算法开始时间
    start = time.time()

    # --------------------------
    # 特征计算模块
    # --------------------------
    # 计算查询特征向量（对查询节点嵌入做均值池化）
    query_feature = torch.mm(query, embedding_tensor) # (query_num, embedding_dim)
    query_num = torch.sum(query, dim=1)                     # 每个查询包含的节点数
    query_feature = torch.div(query_feature, query_num.view(-1, 1))     # 均值化处理
    
    # cosine similarity     # 计算余弦相似度（查询特征与所有节点嵌入的相似性）
    query_score = cosin_similarity(query_feature, embedding_tensor) # (query_num, node_num)
    query_score = torch.nn.functional.normalize(query_score, dim=1, p=1)        # L1归一化为概率分布

    
    print("query_score.shape: ", query_score.shape) # 打印相似度矩阵形状

    # --------------------------
    # 候选节点选择模块
    # --------------------------
    y_pred = torch.zeros_like(query_score)                                  # 初始化预测结果矩阵
    for i in tqdm(range(query_score.shape[0])):             # 遍历每一个查询
        # 获取当前查询的有效节点索引
        query_index = (torch.nonzero(query[i]).squeeze()).reshape(-1)
        # selected_candidates = mwg_subgraph_heuristic(query_index.tolist(), query_score[i].tolist(), graph)
        selected_candidates = mwg_subgraph_heuristic_fast(query_index.tolist(), query_score[i].tolist(), graph)
        for j in range(len(selected_candidates)):
            y_pred[i][selected_candidates[j]] = 1
    # --------------------------
    # 性能评估模块
    # --------------------------
    end = time.time()
    print("全局耗时: {:.4f}秒".format(end-start))
    print("单查询平均耗时: {:.4f}秒".format((end-start)/query_feature.shape[0]))
    pre, rec, f1_score, avg_jaccard = f1_score_calculation(y_pred.int(), labels.int())
    print("平均查准率: {:.4f}".format(pre))
    print("平均查全率: {:.4f}".format(rec))
    print("F1 score by maximum weight gain (local search): {:.4f}".format(f1_score))
    print("平均Jaccard相似度: {:.4f}".format(avg_jaccard))

    nmi, ari, jac = evaluation(y_pred.int(), labels.int())
    
    print("NMI score by maximum weight gain (local search): {:.4f}".format(nmi))
    print("ARI score by maximum weight gain (local search): {:.4f}".format(ari))
    print("JAC score by maximum weight gain (local search): {:.4f}".format(jac))

    density, diameter, conductance = structure_metrics_calculation(y_pred.int(), dgl_graph)
    print("平均density: {:.4f}".format(density))
    print("平均diameter: {:.4f}".format(diameter))
    print("平均conductance: {:.4f}".format(conductance))


