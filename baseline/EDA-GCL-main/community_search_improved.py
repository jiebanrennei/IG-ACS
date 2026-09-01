"""
改进版社区搜索
结合图结构和嵌入相似度
"""

import torch
import numpy as np
import os
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
from collections import deque


def load_data(dataset_name):
    """加载数据集"""
    data_path = f'./dataset/{dataset_name.lower()}/{dataset_name.lower()}_pyg.pt'
    data = torch.load(data_path)
    return data


def load_embeddings(dataset_name):
    """加载嵌入"""
    emb_path = f'./embeddings/{dataset_name}_emb.pt'
    embeddings = torch.load(emb_path)
    return embeddings


def generate_queries(data, num_queries=150, seed=42):
    """生成查询节点和真实社区"""
    np.random.seed(seed)

    # 只从有标签的节点中选择
    valid_mask = data.y >= 0
    valid_indices = torch.where(valid_mask)[0]

    # 按类别分组
    labels = data.y[valid_indices].numpy()
    unique_labels = np.unique(labels)

    queries = []
    ground_truths = []

    for _ in range(num_queries):
        # 随机选择一个类别
        label = np.random.choice(unique_labels)
        # 从该类别中随机选择一个查询节点
        candidates = valid_indices[labels == label]
        query_node = np.random.choice(candidates).item()

        # 真实社区是该类别的所有节点
        community = valid_indices[labels == label].numpy().tolist()

        queries.append([query_node])
        ground_truths.append(community)

    return queries, ground_truths


def build_adjacency_list(edge_index, num_nodes):
    """构建邻接表"""
    adj = {i: [] for i in range(num_nodes)}
    edge_index_np = edge_index.cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        adj[src].append(dst)
        adj[dst].append(src)  # 无向图
    return adj


def community_search_graph_aware(query_nodes, embeddings, adj, data, topk_ratio=0.1, hop=2):
    """
    图感知的社区搜索
    1. 从查询节点开始
    2. BFS 扩展邻居
    3. 用嵌入相似度排序

    Args:
        query_nodes: 查询节点列表
        embeddings: 节点嵌入
        adj: 邻接表
        data: 图数据
        topk_ratio: 返回节点比例
        hop: BFS 跳数

    Returns:
        predicted_community: 预测的社区节点列表
    """
    num_nodes = embeddings.shape[0]
    topk = int(num_nodes * topk_ratio)

    # 计算查询节点的平均嵌入
    query_embs = embeddings[query_nodes]
    query_center = query_embs.mean(dim=0, keepdim=True)

    # BFS 收集候选节点
    candidates = set(query_nodes)
    queue = deque(query_nodes)
    visited = set(query_nodes)

    for _ in range(hop):
        level_size = len(queue)
        for _ in range(level_size):
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    candidates.add(neighbor)
                    queue.append(neighbor)

    # 如果候选节点不够,补充全局最相似的节点
    if len(candidates) < topk:
        similarities = torch.cosine_similarity(embeddings, query_center)
        _, top_indices = torch.topk(similarities, topk)
        candidates.update(top_indices.numpy().tolist())

    # 对候选节点按相似度排序
    candidates = list(candidates)
    candidate_embs = embeddings[candidates]
    similarities = torch.cosine_similarity(candidate_embs, query_center)

    # 选择相似度最高的 topk 个
    _, top_indices = torch.topk(similarities, min(topk, len(candidates)))
    selected = [candidates[i] for i in top_indices.numpy()]

    return selected


def community_search_knn_expansion(query_nodes, embeddings, adj, data, topk_ratio=0.1, k=10):
    """
    KNN 扩展的社区搜索
    1. 对每个查询节点找 K 个最近邻
    2. 迭代扩展
    """
    num_nodes = embeddings.shape[0]
    topk = int(num_nodes * topk_ratio)

    # 计算所有节点间的相似度(用余弦相似度)
    query_embs = embeddings[query_nodes]
    query_center = query_embs.mean(dim=0, keepdim=True)

    # 初始化社区为查询节点
    community = set(query_nodes)

    # 迭代扩展
    for iteration in range(3):
        if len(community) >= topk:
            break

        # 计算当前社区中心的嵌入
        community_embs = embeddings[list(community)]
        center = community_embs.mean(dim=0, keepdim=True)

        # 找最相似的节点
        similarities = torch.cosine_similarity(embeddings, center)
        _, top_indices = torch.topk(similarities, min(k * (iteration + 1), num_nodes))

        # 加入社区
        for idx in top_indices.numpy():
            community.add(idx.item())
            if len(community) >= topk:
                break

    return list(community)[:topk]


def community_search_density(query_nodes, embeddings, adj, data, topk_ratio=0.1, threshold=0.5):
    """
    密度感知的社区搜索
    只选择相似度高于阈值的邻居
    """
    num_nodes = embeddings.shape[0]
    topk = int(num_nodes * topk_ratio)

    query_embs = embeddings[query_nodes]
    query_center = query_embs.mean(dim=0, keepdim=True)

    # 从查询节点开始扩展
    community = set(query_nodes)
    queue = deque(query_nodes)
    visited = set(query_nodes)

    while queue and len(community) < topk:
        node = queue.popleft()

        # 检查邻居
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)

                # 计算相似度
                sim = torch.cosine_similarity(
                    embeddings[neighbor:neighbor+1],
                    query_center
                ).item()

                # 如果相似度高于阈值,加入社区
                if sim >= threshold:
                    community.add(neighbor)
                    queue.append(neighbor)

                    if len(community) >= topk:
                        break

    # 如果不够,补充最相似的节点
    if len(community) < topk:
        similarities = torch.cosine_similarity(embeddings, query_center)
        _, top_indices = torch.topk(similarities, topk)
        community.update(top_indices.numpy().tolist())

    return list(community)[:topk]


def evaluate(pred_community, true_community, num_nodes):
    """评估社区搜索指标"""
    y_true = np.zeros(num_nodes, dtype=int)
    y_pred = np.zeros(num_nodes, dtype=int)

    y_true[true_community] = 1
    y_pred[pred_community] = 1

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    jaccard = intersection / union if union > 0 else 0

    return precision, recall, f1, jaccard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='IMDB', choices=['ACM', 'DBLP', 'IMDB'])
    parser.add_argument('--method', type=str, default='graph',
                       choices=['graph', 'knn', 'density'],
                       help='社区搜索方法')
    parser.add_argument('--topk_ratio', type=float, default=0.2)
    parser.add_argument('--num_queries', type=int, default=150)
    parser.add_argument('--hop', type=int, default=2, help='BFS 跳数(graph方法)')
    parser.add_argument('--k', type=int, default=10, help='KNN 邻居数(knn方法)')
    parser.add_argument('--threshold', type=float, default=0.5, help='相似度阈值(density方法)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"数据集: {args.dataset}")
    print(f"方法: {args.method}")
    print(f"{'='*60}\n")

    # 加载数据
    print("[1/5] 加载数据...")
    data = load_data(args.dataset)
    print(f"  节点数: {data.num_nodes}")

    # 加载嵌入
    print("[2/5] 加载嵌入...")
    embeddings = load_embeddings(args.dataset)

    # 构建邻接表
    print("[3/5] 构建邻接表...")
    adj = build_adjacency_list(data.edge_index, data.num_nodes)

    # 生成查询
    print("[4/5] 生成查询...")
    queries, ground_truths = generate_queries(data, args.num_queries)

    # 社区搜索和评估
    print("[5/5] 社区搜索和评估...")
    precisions = []
    recalls = []
    f1s = []
    jaccards = []

    for i, (query, gt) in enumerate(zip(queries, ground_truths)):
        if args.method == 'graph':
            pred = community_search_graph_aware(
                query, embeddings, adj, data,
                args.topk_ratio, args.hop
            )
        elif args.method == 'knn':
            pred = community_search_knn_expansion(
                query, embeddings, adj, data,
                args.topk_ratio, args.k
            )
        elif args.method == 'density':
            pred = community_search_density(
                query, embeddings, adj, data,
                args.topk_ratio, args.threshold
            )

        p, r, f1, jac = evaluate(pred, gt, data.num_nodes)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        jaccards.append(jac)

        if (i + 1) % 30 == 0:
            print(f"  已处理 {i+1}/{len(queries)} 个查询")

    # 输出结果
    print(f"\n{'='*60}")
    print("社区搜索结果:")
    print(f"{'='*60}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Jaccard:   {np.mean(jaccards):.4f} ± {np.std(jaccards):.4f}")
    print(f"{'='*60}\n")

    # 保存结果
    result_file = f'./results/{args.dataset}_{args.method}_search.txt'
    os.makedirs('./results', exist_ok=True)
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Top-k ratio: {args.topk_ratio}\n")
        if args.method == 'graph':
            f.write(f"Hop: {args.hop}\n")
        elif args.method == 'knn':
            f.write(f"K: {args.k}\n")
        elif args.method == 'density':
            f.write(f"Threshold: {args.threshold}\n")
        f.write(f"\n")
        f.write(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}\n")
        f.write(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}\n")
        f.write(f"F1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n")
        f.write(f"Jaccard:   {np.mean(jaccards):.4f} ± {np.std(jaccards):.4f}\n")
    print(f"结果已保存到: {result_file}\n")


if __name__ == '__main__':
    main()
