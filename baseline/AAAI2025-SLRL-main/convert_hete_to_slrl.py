"""
将异构图数据集转换为 SLRL 格式
SLRL 需要：
1. 边文件：每行一条边 (u v)
2. 社区文件：每行一个社区（节点列表）
3. 种子节点和社区索引文件
"""

import os
import sys
import numpy as np
from scipy import sparse as sp
import argparse

# 添加共享模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.dataset_utils import HETE_DATASETS


def load_hete_dataset(dataset_name, data_root='../../datasets'):
    """加载异构图数据集"""
    print(f"\n{'='*60}")
    print(f"加载 {dataset_name} 数据集")
    print(f"{'='*60}\n")

    # 数据集名称映射（支持别名）
    name_map = {
        'IMDB_NEW': 'IMDB',
        'IMDB': 'IMDB',
        'ACM': 'ACM',
        'DBLP': 'DBLP'
    }

    dataset_name_upper = dataset_name.upper()
    if dataset_name_upper in name_map:
        dataset_name_upper = name_map[dataset_name_upper]

    if dataset_name_upper not in HETE_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(HETE_DATASETS.keys())}")

    cfg = HETE_DATASETS[dataset_name_upper]
    data_dir = os.path.join(data_root, cfg['dir'])

    # 1. 加载标签
    labels = np.load(os.path.join(data_dir, cfg['labels']))
    print(f"标签数: {len(labels)}")
    print(f"类别数: {len(np.unique(labels))}")

    # 2. 加载邻接矩阵（合并所有 meta-path）
    print("\n加载边...")

    # 获取所有 meta-path 文件
    meta_path_files = []
    for mp_file in cfg['meta_paths']:
        mp_path = os.path.join(data_dir, mp_file)
        if os.path.exists(mp_path):
            meta_path_files.append(mp_path)

    # 如果没有找到 meta-path 文件，使用 adj.npz
    if not meta_path_files:
        adj_file = os.path.join(data_dir, 'adj.npz')
        if os.path.exists(adj_file):
            meta_path_files.append(adj_file)

    print(f"找到 {len(meta_path_files)} 个 meta-path 文件")

    # 合并所有边
    all_edges = []
    for mp_file in meta_path_files:
        print(f"  加载: {os.path.basename(mp_file)}")
        adj = sp.load_npz(mp_file)
        # 转换为边列表
        adj_coo = adj.tocoo()
        edges = np.column_stack([adj_coo.row, adj_coo.col])
        # 只保留上三角（无向图）
        edges = edges[edges[:, 0] < edges[:, 1]]
        all_edges.append(edges)
        print(f"    边数: {len(edges)}")

    # 合并并去重
    all_edges = np.vstack(all_edges)
    all_edges = np.unique(all_edges, axis=0)
    print(f"\n总边数（去重后）: {len(all_edges)}")

    # 3. 获取总节点数并重新标记节点（确保连续）
    unique_nodes = np.unique(all_edges.flatten())
    num_nodes = len(unique_nodes)

    # 创建节点映射（旧ID -> 新ID）
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}

    # 重新映射边
    all_edges_remapped = np.array([[node_mapping[u], node_mapping[v]] for u, v in all_edges])

    # 重新映射标签
    labels_remapped = np.full(num_nodes, -1, dtype=np.int64)
    for old_id, label in enumerate(labels):
        if old_id in node_mapping:
            labels_remapped[node_mapping[old_id]] = label

    print(f"总节点数: {num_nodes}")
    print(f"有标签的节点数: {np.sum(labels_remapped >= 0)}")

    return all_edges_remapped, labels_remapped, num_nodes


def convert_to_slrl_format(dataset_name, output_dir, data_root='../../datasets'):
    """转换为 SLRL 格式"""

    # 加载数据
    edges, labels, num_nodes = load_hete_dataset(dataset_name, data_root)

    # 创建输出目录
    output_path = os.path.join(output_dir, dataset_name)
    os.makedirs(output_path, exist_ok=True)

    # 1. 保存边文件
    edge_file = os.path.join(output_path, f'{dataset_name}-1.90.ungraph.txt')
    with open(edge_file, 'w') as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")
    print(f"\n边文件已保存: {edge_file}")
    print(f"  边数: {len(edges)}")

    # 2. 按标签分组生成社区
    print("\n生成社区...")
    communities = {}
    for node_id, label in enumerate(labels):
        if label not in communities:
            communities[label] = []
        communities[label].append(node_id)

    # 过滤掉太小的社区（少于5个节点）
    communities = {k: v for k, v in communities.items() if len(v) >= 5}
    print(f"  社区数（过滤后）: {len(communities)}")
    print(f"  平均社区大小: {np.mean([len(v) for v in communities.values()]):.2f}")

    # 保存社区文件
    com_file = os.path.join(output_path, f'{dataset_name}-1.90.cmty.txt')
    with open(com_file, 'w') as f:
        for label, nodes in communities.items():
            f.write(' '.join(map(str, nodes)) + '\n')
    print(f"社区文件已保存: {com_file}")

    # 3. 生成种子节点和社区索引
    # 从每个社区中选择第一个节点作为种子
    seeds = []
    com_indices = []
    for idx, (label, nodes) in enumerate(communities.items()):
        seeds.append([nodes[0]])  # 种子节点
        com_indices.append([idx])  # 社区索引

    return seeds, com_indices


def generate_seed_files(all_seeds, all_com_indices, output_dir):
    """生成种子节点和社区索引文件（所有数据集）"""
    # 保存 seed12 文件
    seed_file = os.path.join(output_dir, 'seed12')
    with open(seed_file, 'w') as f:
        for dataset_seeds in all_seeds:
            for seed_list in dataset_seeds:
                f.write(' '.join(map(str, seed_list)) + '\n')
    print(f"\n种子文件已保存: {seed_file}")

    # 保存 com_index12 文件
    com_file = os.path.join(output_dir, 'com_index12')
    with open(com_file, 'w') as f:
        for dataset_com_indices in all_com_indices:
            for com_list in dataset_com_indices:
                f.write(' '.join(map(str, com_list)) + '\n')
    print(f"社区索引文件已保存: {com_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['acm', 'dblp', 'imdb_new'],
                       help='数据集列表')
    parser.add_argument('--output_dir', type=str, default='./datasets',
                       help='输出目录')
    parser.add_argument('--data_root', type=str, default='../../datasets',
                       help='数据根目录')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"转换异构图数据集到 SLRL 格式")
    print(f"{'='*60}")
    print(f"数据集: {args.datasets}")
    print(f"输出目录: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_seeds = []
    all_com_indices = []

    for dataset_name in args.datasets:
        seeds, com_indices = convert_to_slrl_format(
            dataset_name,
            args.output_dir,
            args.data_root
        )
        all_seeds.append(seeds)
        all_com_indices.append(com_indices)

    # 生成统一的种子文件
    generate_seed_files(all_seeds, all_com_indices, args.output_dir)

    print(f"\n{'='*60}")
    print("转换完成!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
