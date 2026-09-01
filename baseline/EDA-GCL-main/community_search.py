"""
基于 EDA-GCL 嵌入的社区搜索
使用相似度扩展算法
"""

import torch
import numpy as np
import os
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import argparse


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


def community_search(query_nodes, embeddings, data, topk_ratio=0.1):
    """
    基于嵌入相似度的社区搜索

    Args:
        query_nodes: 查询节点列表
        embeddings: 节点嵌入 [num_nodes, dim]
        data: 图数据
        topk_ratio: 返回前 topk_ratio * num_nodes 个节点

    Returns:
        predicted_community: 预测的社区节点列表
    """
    num_nodes = embeddings.shape[0]
    topk = int(num_nodes * topk_ratio)

    # 计算查询节点的平均嵌入
    query_embs = embeddings[query_nodes]
    query_center = query_embs.mean(dim=0, keepdim=True)

    # 计算所有节点与查询中心的相似度
    similarities = torch.cosine_similarity(embeddings, query_center)

    # 选择相似度最高的 topk 个节点
    _, top_indices = torch.topk(similarities, topk)

    return top_indices.numpy().tolist()


def evaluate(pred_community, true_community, num_nodes):
    """评估社区搜索指标"""
    # 转换为二值向量
    y_true = np.zeros(num_nodes, dtype=int)
    y_pred = np.zeros(num_nodes, dtype=int)

    y_true[true_community] = 1
    y_pred[pred_community] = 1

    # 计算指标
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Jaccard
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    jaccard = intersection / union if union > 0 else 0

    return precision, recall, f1, jaccard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='IMDB', choices=['ACM', 'DBLP', 'IMDB'])
    parser.add_argument('--topk_ratio', type=float, default=0.1, help='返回节点比例')
    parser.add_argument('--num_queries', type=int, default=150, help='查询数量')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"数据集: {args.dataset}")
    print(f"{'='*60}\n")

    # 加载数据
    print("[1/4] 加载数据...")
    data = load_data(args.dataset)
    print(f"  节点数: {data.num_nodes}")
    print(f"  边数: {data.num_edges}")

    # 加载嵌入
    print("[2/4] 加载嵌入...")
    embeddings = load_embeddings(args.dataset)
    print(f"  嵌入形状: {embeddings.shape}")

    # 生成查询
    print("[3/4] 生成查询...")
    queries, ground_truths = generate_queries(data, args.num_queries)
    print(f"  查询数: {len(queries)}")

    # 社区搜索和评估
    print("[4/4] 社区搜索和评估...")
    precisions = []
    recalls = []
    f1s = []
    jaccards = []

    for i, (query, gt) in enumerate(zip(queries, ground_truths)):
        pred = community_search(query, embeddings, data, args.topk_ratio)
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
    result_file = f'./results/{args.dataset}_community_search.txt'
    os.makedirs('./results', exist_ok=True)
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Top-k ratio: {args.topk_ratio}\n")
        f.write(f"Num queries: {args.num_queries}\n")
        f.write(f"\n")
        f.write(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}\n")
        f.write(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}\n")
        f.write(f"F1:        {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n")
        f.write(f"Jaccard:   {np.mean(jaccards):.4f} ± {np.std(jaccards):.4f}\n")
    print(f"结果已保存到: {result_file}\n")


if __name__ == '__main__':
    main()
