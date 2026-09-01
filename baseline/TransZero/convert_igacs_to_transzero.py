"""
将 IG-ACS 的异构图数据集转换为 TransZero 格式
支持: ACM, DBLP, IMDB_NEW
"""

import os
import sys
import torch
import numpy as np
import scipy.sparse as sp
from pathlib import Path

# 添加父目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_cs_dataset, CS_DATASETS


def convert_to_transzero_format(dataset_name, output_dir):
    """
    转换数据集到 TransZero 格式

    Args:
        dataset_name: 数据集名称 (ACM, DBLP, IMDB_NEW)
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"转换 {dataset_name} 到 TransZero 格式")
    print(f"{'='*60}\n")

    # 加载数据（合并所有 meta-path 为同构图）
    dataset = get_cs_dataset(
        './datasets/',
        dataset_name,
        meta_path='all',  # 合并所有 meta-path
        cs_full_graph=True
    )
    data = dataset[0]

    # 获取基本信息
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    num_classes = data.y.max().item() + 1

    print(f"节点数: {num_nodes}")
    print(f"边数: {num_edges}")
    print(f"类别数: {num_classes}")

    # 创建输出目录
    output_path = Path(output_dir) / dataset_name.lower()
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 保存 .pt 文件 (邻接矩阵 + 特征 + 标签)
    print(f"\n[1/4] 保存 .pt 文件...")

    # 构建邻接矩阵 (COO 格式)
    edge_index = data.edge_index
    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(
        edge_index,
        values,
        (num_nodes, num_nodes)
    ).coalesce()

    # 保存为 .pt 文件
    pt_data = [adj, data.x, data.y]
    pt_file = output_path / f"{dataset_name.lower()}.pt"
    torch.save(pt_data, pt_file)
    print(f"  保存到: {pt_file}")

    # 2. 保存 .edges 文件
    print(f"\n[2/4] 保存 .edges 文件...")
    edges_file = output_path / f"{dataset_name.lower()}.edges"
    with open(edges_file, 'w') as f:
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            f.write(f"{src} {dst}\n")
    print(f"  保存到: {edges_file}")

    # 3. 生成查询和真实社区
    print(f"\n[3/4] 生成查询节点和真实社区...")

    # 按类别构建社区
    labels = data.y.numpy()
    communities = {}
    for node_id, label in enumerate(labels):
        if label not in communities:
            communities[label] = []
        communities[label].append(node_id)

    print(f"  社区数量: {len(communities)}")
    for label, members in list(communities.items())[:5]:
        print(f"    类别 {label}: {len(members)} 个节点")

    # 生成 150 个查询
    np.random.seed(42)
    num_queries = 150

    selected_queries = []
    ground_truth = []

    for i in range(num_queries):
        # 随机选择一个类别
        selected_class = np.random.choice(list(communities.keys()))
        community_members = communities[selected_class]

        # 随机选择 1 个查询节点
        num_query_nodes = 1
        query_nodes = np.random.choice(community_members, num_query_nodes, replace=False).tolist()

        selected_queries.append(query_nodes)
        ground_truth.append(community_members)

    # 4. 保存 .query 和 .gt 文件
    print(f"\n[4/4] 保存 .query 和 .gt 文件...")

    query_file = output_path / f"{dataset_name.lower()}.query"
    gt_file = output_path / f"{dataset_name.lower()}.gt"

    with open(query_file, 'w') as f:
        for query_nodes in selected_queries:
            f.write(' '.join(map(str, query_nodes)) + '\n')
    print(f"  保存到: {query_file}")

    with open(gt_file, 'w') as f:
        for community in ground_truth:
            f.write(' '.join(map(str, community)) + '\n')
    print(f"  保存到: {gt_file}")

    print(f"\n✅ {dataset_name} 转换完成！")
    print(f"输出目录: {output_path}")
    print(f"\n文件列表:")
    for file in output_path.iterdir():
        print(f"  - {file.name}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='转换数据集到 TransZero 格式')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['ACM', 'DBLP', 'IMDB_NEW', 'all'],
                       help='要转换的数据集')
    parser.add_argument('--output_dir', type=str, default='baseline/TransZero/dataset',
                       help='输出目录')

    args = parser.parse_args()

    if args.dataset == 'all':
        datasets = ['ACM', 'DBLP', 'IMDB_NEW']
    else:
        datasets = [args.dataset]

    for dataset_name in datasets:
        convert_to_transzero_format(dataset_name, args.output_dir)

    print(f"\n{'='*60}")
    print("所有数据集转换完成！")
    print(f"{'='*60}\n")
