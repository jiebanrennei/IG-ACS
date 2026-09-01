"""
将 SNAP 同构图数据集转换为 TransZero 格式
支持: com-Amazon, com-DBLP, com-Youtube, com-LiveJournal, com-Twitter
独立版本,不依赖 utils.py
"""

import os
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from collections import defaultdict

# SNAP 数据集配置
SNAP_DATASETS = {
    'com-Amazon': {
        'edge_file': 'com-amazon.ungraph.txt',
        'cmty_file': 'com-amazon.top5000.cmty.txt',
        'max_communities': 5000,
    },
    'com-DBLP': {
        'edge_file': 'com-dblp.ungraph.txt',
        'cmty_file': 'com-dblp.top5000.cmty.txt',
        'max_communities': 5000,
    },
    'com-Youtube': {
        'edge_file': 'com-youtube.ungraph.txt',
        'cmty_file': 'com-youtube.top5000.cmty.txt',
        'max_communities': 5000,
    },
    'com-LiveJournal': {
        'edge_file': 'com-lj.ungraph.txt',
        'cmty_file': 'com-lj.top5000.cmty.txt',
        'max_communities': 5000,
    },
    'com-Twitter': {
        'edge_file': 'com-twitter.ungraph.txt',
        'cmty_file': 'com-twitter.cmty.txt',
        'max_communities': None,
    },
}


def load_snap_edges(edge_file):
    """加载 SNAP 格式的边文件"""
    edges = []
    max_node = 0
    with open(edge_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
                max_node = max(max_node, u, v)
    return edges, max_node + 1


def load_snap_communities(cmty_file, max_communities=None):
    """加载 SNAP 格式的社区文件"""
    communities = []
    with open(cmty_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                members = [int(x) for x in parts]
                communities.append(members)

    # 只保留大小 >= 3 的社区
    communities = [c for c in communities if len(c) >= 3]

    # 限制社区数量
    if max_communities and len(communities) > max_communities:
        # 按大小排序,取最大的
        communities.sort(key=len, reverse=True)
        communities = communities[:max_communities]

    return communities


def build_adjacency_matrix(edges, num_nodes):
    """构建邻接矩阵"""
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    # 对称化
    rows_sym = rows + cols
    cols_sym = cols + rows

    adj = sp.coo_matrix(
        (np.ones(len(rows_sym)), (rows_sym, cols_sym)),
        shape=(num_nodes, num_nodes)
    )
    adj = adj.tocsr()
    # 二值化
    adj.data = np.ones_like(adj.data)
    # 去自环
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def generate_degree_features(adj, num_nodes, feat_dim=128):
    """基于度数生成节点特征"""
    degrees = np.array(adj.sum(axis=1)).flatten()
    log_degrees = np.log1p(degrees)

    # 归一化
    max_deg = log_degrees.max()
    if max_deg > 0:
        log_degrees = log_degrees / max_deg

    # 生成多维度特征 (度数 + 随机投影)
    np.random.seed(42)
    features = np.zeros((num_nodes, feat_dim))
    features[:, 0] = log_degrees

    # 其他维度使用随机投影
    if feat_dim > 1:
        projection = np.random.randn(1, feat_dim - 1)
        features[:, 1:] = log_degrees.reshape(-1, 1) * projection

    # L2 归一化
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    features = features / norms

    return sp.csr_matrix(features)


def assign_node_labels(communities, num_nodes):
    """为节点分配标签 (基于社区成员关系)"""
    labels = np.full(num_nodes, -1, dtype=np.int64)

    # 统计每个节点出现在多少个社区中
    node_community_count = defaultdict(list)
    for comm_idx, members in enumerate(communities):
        for node in members:
            node_community_count[node].append(comm_idx)

    # 为每个节点分配它出现的第一个社区的标签
    for node, comm_indices in node_community_count.items():
        labels[node] = comm_indices[0]

    # 没有社区的节点标记为 0
    labels[labels == -1] = 0

    return labels


def convert_snap_to_transzero(dataset_name, output_dir, data_root='../../datasets'):
    """
    转换 SNAP 数据集到 TransZero 格式

    Args:
        dataset_name: 数据集名称 (com-Amazon, com-DBLP, etc.)
        output_dir: 输出目录
        data_root: 数据根目录
    """
    print(f"\n{'='*60}")
    print(f"转换 {dataset_name} 到 TransZero 格式")
    print(f"{'='*60}\n")

    if dataset_name not in SNAP_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(SNAP_DATASETS.keys())}")

    cfg = SNAP_DATASETS[dataset_name]
    base = os.path.join(data_root, dataset_name)

    # 1. 加载边
    print(f"[1/5] 加载边...")
    edge_file = os.path.join(base, cfg['edge_file'])
    edges, num_nodes = load_snap_edges(edge_file)
    print(f"  节点数: {num_nodes}")
    print(f"  边数: {len(edges)}")

    # 2. 构建邻接矩阵
    print(f"[2/5] 构建邻接矩阵...")
    adj = build_adjacency_matrix(edges, num_nodes)
    print(f"  邻接矩阵非零元: {adj.nnz}")

    # 3. 生成特征
    print(f"[3/5] 生成节点特征...")
    features = generate_degree_features(adj, num_nodes, feat_dim=128)
    print(f"  特征形状: {features.shape}")

    # 4. 加载社区
    print(f"[4/5] 加载社区...")
    cmty_file = os.path.join(base, cfg['cmty_file'])
    communities = load_snap_communities(cmty_file, cfg['max_communities'])
    print(f"  社区数量: {len(communities)}")

    # 分配标签
    labels = assign_node_labels(communities, num_nodes)
    num_classes = len(set(labels))
    print(f"  类别数: {num_classes}")

    # 5. 保存
    print(f"[5/5] 保存到 TransZero 格式...")
    output_path = Path(output_dir) / dataset_name.lower().replace('-', '_')
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存 final_meta_path.npz (邻接矩阵)
    adj_file = output_path / "final_meta_path.npz"
    sp.save_npz(adj_file, adj.tocoo())
    print(f"  保存: {adj_file}")

    # 保存 feat.npz (特征)
    feat_file = output_path / "feat.npz"
    sp.save_npz(feat_file, features)
    print(f"  保存: {feat_file}")

    # 保存 labels.npy
    labels_file = output_path / "labels.npy"
    np.save(labels_file, labels)
    print(f"  保存: {labels_file}")

    # 生成查询节点
    print(f"  生成查询节点...")
    np.random.seed(42)
    num_queries = 150

    # 只从有标签的节点中选择
    labeled_nodes = np.where(labels > 0)[0]
    if len(labeled_nodes) < num_queries:
        num_queries = len(labeled_nodes)

    query_nodes = np.random.choice(labeled_nodes, num_queries, replace=False)

    query_file = output_path / f"{dataset_name.lower().replace('-', '_')}.query"
    with open(query_file, 'w') as f:
        for node in query_nodes:
            f.write(f"{node}\n")
    print(f"  保存: {query_file}")

    # 生成真实社区
    gt_file = output_path / f"{dataset_name.lower().replace('-', '_')}.gt"
    with open(gt_file, 'w') as f:
        for query_node in query_nodes:
            # 找到包含该查询节点的社区
            for comm_idx, members in enumerate(communities):
                if query_node in members:
                    f.write(' '.join(map(str, members)) + '\n')
                    break
            else:
                # 如果没找到,写空
                f.write(f"{query_node}\n")
    print(f"  保存: {gt_file}")

    print(f"\n✅ {dataset_name} 转换完成!")
    print(f"输出目录: {output_path}")
    print(f"\n文件列表:")
    for file in output_path.iterdir():
        print(f"  - {file.name}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='转换 SNAP 数据集到 TransZero 格式')
    parser.add_argument('--dataset', type=str, default='com-Amazon',
                       choices=['com-Amazon', 'com-DBLP', 'com-Youtube',
                               'com-LiveJournal', 'com-Twitter', 'all'],
                       help='要转换的数据集')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='输出目录')
    parser.add_argument('--data_root', type=str, default='../../datasets',
                       help='数据根目录')

    args = parser.parse_args()

    if args.dataset == 'all':
        datasets = ['com-Amazon', 'com-DBLP', 'com-Youtube']
    else:
        datasets = [args.dataset]

    for dataset_name in datasets:
        try:
            convert_snap_to_transzero(dataset_name, args.output_dir, args.data_root)
        except Exception as e:
            print(f"\n❌ {dataset_name} 转换失败: {e}\n")

    print(f"\n{'='*60}")
    print("转换完成!")
    print(f"{'='*60}\n")
