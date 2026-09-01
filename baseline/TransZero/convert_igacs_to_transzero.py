"""
将 IG-ACS 的异构图数据集转换为 TransZero 格式
支持: ACM, DBLP, IMDB_NEW
独立版本,不依赖 utils.py
"""

import os
import torch
import numpy as np
import scipy.sparse as sp
from pathlib import Path

# 数据集配置
CS_DATASETS = {
    'ACM': {
        'dir': 'acm',
        'feat': 'p_feat.npz',
        'feature_norm': 'row_sum',
        'meta_paths': ['pap.npz', 'psp.npz'],
    },
    'DBLP': {
        'dir': 'dblp',
        'feat': 'a_feat.npz',
        'feature_norm': 'row_sum',
        'meta_paths': ['apa.npz', 'apcpa.npz', 'aptpa.npz'],
    },
    'IMDB_NEW': {
        'dir': 'imdb_new',
        'feat': 'm_feat.npz',
        'feature_norm': 'l2',
        'meta_paths': ['mam.npz', 'mdm.npz'],
    },
}


def _normalize_cs_features(x, mode):
    """归一化特征"""
    if mode == 'row_sum':
        denom = x.sum(dim=1, keepdim=True).clamp_min(1e-12)
    elif mode == 'l2':
        denom = x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    else:
        raise ValueError(f"Unknown normalization: {mode}")
    return x / denom


def _binarize_symmetric(adj_sp):
    """二值化 + 对称化 + 去自环"""
    a = (adj_sp > 0).astype(np.uint8).tocsr()
    a = a.maximum(a.T).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def convert_to_transzero_format(dataset_name, output_dir, data_root='../../datasets'):
    """
    转换数据集到 TransZero 格式

    Args:
        dataset_name: 数据集名称 (ACM, DBLP, IMDB_NEW)
        output_dir: 输出目录
        data_root: 数据根目录
    """
    print(f"\n{'='*60}")
    print(f"转换 {dataset_name} 到 TransZero 格式")
    print(f"{'='*60}\n")

    if dataset_name not in CS_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(CS_DATASETS.keys())}")

    cfg = CS_DATASETS[dataset_name]
    base = os.path.join(data_root, cfg['dir'])

    # 1. 加载特征
    print(f"[1/4] 加载特征...")
    feat_sp = sp.load_npz(os.path.join(base, cfg['feat']))
    x = torch.from_numpy(feat_sp.toarray()).float()
    x = _normalize_cs_features(x, cfg['feature_norm'])
    print(f"  特征形状: {x.shape}")

    # 2. 加载标签
    print(f"[2/4] 加载标签...")
    labels = np.load(os.path.join(base, 'labels.npy'))
    y = torch.from_numpy(labels.astype(np.int64))
    print(f"  节点数: {len(labels)}")
    print(f"  类别数: {y.max().item() + 1}")

    # 3. 合并所有 meta-path
    print(f"[3/4] 合并 meta-paths: {cfg['meta_paths']}...")
    adj = None
    for mp in cfg['meta_paths']:
        mp_path = os.path.join(base, mp)
        print(f"  加载 {mp}...")
        m = (sp.load_npz(mp_path) > 0).astype(np.uint8)
        print(f"    边数: {m.nnz}")
        adj = m if adj is None else adj.maximum(m)

    adj = _binarize_symmetric(adj)
    num_edges = adj.nnz
    print(f"  合并后边数: {num_edges}")

    # 转换为 edge_index
    coo = adj.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)

    num_nodes = len(labels)

    # 4. 创建输出目录并保存
    print(f"[4/4] 保存到 TransZero 格式...")
    output_path = Path(output_dir) / dataset_name.lower()
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存 .pt 文件 (邻接矩阵 + 特征 + 标签)
    print(f"  保存 .pt 文件...")
    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    adj_sparse = torch.sparse_coo_tensor(
        edge_index,
        values,
        (num_nodes, num_nodes)
    ).coalesce()

    pt_data = [adj_sparse, x, y]
    pt_file = output_path / f"{dataset_name.lower()}.pt"
    torch.save(pt_data, pt_file)
    print(f"    保存到: {pt_file}")

    # 保存 .edges 文件
    print(f"  保存 .edges 文件...")
    edges_file = output_path / f"{dataset_name.lower()}.edges"
    with open(edges_file, 'w') as f:
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            f.write(f"{src} {dst}\n")
    print(f"    保存到: {edges_file}")

    # 生成查询节点和真实社区
    print(f"  生成查询和真实社区...")
    communities = {}
    for node_id, label in enumerate(labels):
        if label not in communities:
            communities[label] = []
        communities[label].append(node_id)

    print(f"    社区数量: {len(communities)}")
    for label, members in list(communities.items())[:5]:
        print(f"      类别 {label}: {len(members)} 个节点")

    # 生成 150 个查询
    np.random.seed(42)
    num_queries = 150

    selected_queries = []
    ground_truth = []

    for i in range(num_queries):
        selected_class = np.random.choice(list(communities.keys()))
        community_members = communities[selected_class]
        query_nodes = np.random.choice(community_members, 1, replace=False).tolist()
        selected_queries.append(query_nodes)
        ground_truth.append(community_members)

    # 保存 .query 和 .gt 文件
    query_file = output_path / f"{dataset_name.lower()}.query"
    gt_file = output_path / f"{dataset_name.lower()}.gt"

    with open(query_file, 'w') as f:
        for query_nodes in selected_queries:
            f.write(' '.join(map(str, query_nodes)) + '\n')
    print(f"    保存到: {query_file}")

    with open(gt_file, 'w') as f:
        for community in ground_truth:
            f.write(' '.join(map(str, community)) + '\n')
    print(f"    保存到: {gt_file}")

    print(f"\n✅ {dataset_name} 转换完成!")
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
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='输出目录')
    parser.add_argument('--data_root', type=str, default='../../datasets',
                       help='数据根目录')

    args = parser.parse_args()

    if args.dataset == 'all':
        datasets = ['ACM', 'DBLP', 'IMDB_NEW']
    else:
        datasets = [args.dataset]

    for dataset_name in datasets:
        convert_to_transzero_format(dataset_name, args.output_dir, args.data_root)

    print(f"\n{'='*60}")
    print("所有数据集转换完成!")
    print(f"{'='*60}\n")
