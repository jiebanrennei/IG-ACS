"""
将 IG-ACS 的异构图数据集转换为 EDA-GCL 格式
使用 TransZero 的异构图转换方法:
1. 分别加载不同类型的节点特征
2. 补齐维度(padding)
3. 按节点类型顺序拼接
4. 缺失特征的节点类型用单位矩阵
"""

import os
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from torch_geometric.data import Data

# 数据集配置
HETE_DATASETS = {
    'ACM': {
        'dir': 'acm',
        'node_types': ['paper', 'author', 'subject'],
        'features': {
            'paper': 'p_feat.npz',
            'author': 'a_feat.npz',
            # subject 没有特征,用单位矩阵
        },
        'meta_paths': ['pap.npz', 'psp.npz'],
        'labels': 'labels.npy',
    },
    'DBLP': {
        'dir': 'dblp',
        'node_types': ['author', 'paper', 'term', 'conference'],
        'features': {
            'author': 'a_feat.npz',
            'paper': 'p_feat.npz',
            'term': 't_feat.npz',
            # conference 没有特征,用单位矩阵
        },
        'meta_paths': ['apa.npz', 'apcpa.npz', 'aptpa.npz'],
        'labels': 'labels.npy',
    },
    'IMDB': {
        'dir': 'imdb_new',
        'node_types': ['movie', 'actor', 'director'],
        'features': {
            'movie': 'm_feat.npz',
            # actor 和 director 没有特征,用单位矩阵
        },
        'meta_paths': ['mam.npz', 'mdm.npz'],
        'labels': 'labels.npy',
    },
}


def pad_features(feat, target_dim):
    """补齐特征维度"""
    if feat.shape[1] >= target_dim:
        return feat
    padding = torch.zeros(feat.shape[0], target_dim - feat.shape[1])
    return torch.cat([feat, padding], dim=1)


def convert_hete_to_homo(dataset_name, output_dir, data_root='../../datasets'):
    """
    转换异构图到同构图格式

    Args:
        dataset_name: 数据集名称 (ACM, DBLP, IMDB)
        output_dir: 输出目录
        data_root: 数据根目录
    """
    print(f"\n{'='*60}")
    print(f"转换 {dataset_name} 异构图到同构图")
    print(f"{'='*60}\n")

    if dataset_name not in HETE_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cfg = HETE_DATASETS[dataset_name]
    base = os.path.join(data_root, cfg['dir'])

    # 1. 加载标签
    print(f"[1/5] 加载标签...")
    labels = np.load(os.path.join(base, cfg['labels']))
    print(f"  原始标签数: {len(labels)}")

    # 从邻接矩阵获取总节点数
    adj_temp = sp.load_npz(os.path.join(base, cfg['meta_paths'][0]))
    num_nodes = adj_temp.shape[0]
    print(f"  总节点数: {num_nodes}")

    # 扩展标签到所有节点(无标签的填-1)
    if len(labels) < num_nodes:
        extended_labels = np.full(num_nodes, -1, dtype=np.int64)
        extended_labels[:len(labels)] = labels
        labels = extended_labels
        print(f"  扩展后标签数: {len(labels)} (其中 {np.sum(labels >= 0)} 个有标签)")

    # 2. 加载各类型节点特征
    print(f"[2/5] 加载节点特征...")
    features_list = []
    node_counts = {}

    for node_type in cfg['node_types']:
        if node_type in cfg['features']:
            feat_file = cfg['features'][node_type]
            feat_path = os.path.join(base, feat_file)

            # 尝试加载为稀疏矩阵或稠密数组
            try:
                feat_sp = sp.load_npz(feat_path)
                feat = torch.from_numpy(feat_sp.toarray()).float()
            except:
                # 如果是稠密数组
                feat_data = np.load(feat_path)
                feat = torch.from_numpy(feat_data).float()

            print(f"  {node_type}: {feat.shape}")
            features_list.append((node_type, feat))
            node_counts[node_type] = feat.shape[0]
        else:
            # 没有特征的节点类型,先用 None 占位,后面用单位矩阵
            features_list.append((node_type, None))

    # 3. 计算总节点数和各类型节点数
    print(f"[3/5] 计算节点分布...")

    # 使用步骤1加载的邻接矩阵
    total_nodes_in_graph = num_nodes

    # 计算有特征的节点总数
    total_feat_nodes = sum(feat.shape[0] for _, feat in features_list if feat is not None)

    # 剩余节点分配给没有特征的节点类型
    remaining_nodes = total_nodes_in_graph - total_feat_nodes
    print(f"  图总节点数: {total_nodes_in_graph}")
    print(f"  有特征节点: {total_feat_nodes}")
    print(f"  无特征节点(用单位矩阵): {remaining_nodes}")

    # 4. 补齐维度并拼接特征
    print(f"[4/5] 补齐维度并拼接特征...")
    max_dim = max(feat.shape[1] for _, feat in features_list if feat is not None)
    print(f"  最大特征维度: {max_dim}")

    padded_features = []
    node_type_ranges = {}
    current_idx = 0

    for node_type, feat in features_list:
        if feat is not None:
            padded = pad_features(feat, max_dim)
            padded_features.append(padded)
            node_type_ranges[node_type] = (current_idx, current_idx + feat.shape[0])
            current_idx += feat.shape[0]
        else:
            # 用单位矩阵
            identity = torch.eye(remaining_nodes, max_dim)
            padded_features.append(identity)
            node_type_ranges[node_type] = (current_idx, current_idx + remaining_nodes)
            current_idx += remaining_nodes

    # 拼接所有特征
    all_features = torch.cat(padded_features, dim=0)
    print(f"  最终特征形状: {all_features.shape}")

    # 5. 合并所有 meta-path
    print(f"[5/5] 合并 meta-paths...")
    adj = None
    for mp in cfg['meta_paths']:
        mp_path = os.path.join(base, mp)
        print(f"  加载 {mp}...")
        m = (sp.load_npz(mp_path) > 0).astype(np.uint8)
        print(f"    边数: {m.nnz}")
        adj = m if adj is None else adj.maximum(m)

    # 二值化 + 对称化 + 去自环
    adj = (adj > 0).astype(np.uint8).tocsr()
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()

    # 转换为 edge_index
    coo = adj.tocoo()
    edge_index = torch.stack([
        torch.from_numpy(coo.row).long(),
        torch.from_numpy(coo.col).long()
    ], dim=0)

    print(f"  合并后边数: {edge_index.shape[1]}")

    # 6. 创建 PyG Data 对象
    data = Data(
        x=all_features,
        edge_index=edge_index,
        y=torch.from_numpy(labels).long()
    )

    # 7. 保存
    output_path = Path(output_dir) / dataset_name.lower()
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存为 PyTorch 文件
    output_file = output_path / f"{dataset_name.lower()}_pyg.pt"
    torch.save(data, output_file)
    print(f"\n[OK] 保存到: {output_file}")

    # 打印节点类型分布
    print(f"\n节点类型分布:")
    for node_type, (start, end) in node_type_ranges.items():
        print(f"  {node_type}: [{start}, {end})")

    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['ACM', 'DBLP', 'IMDB', 'all'])
    parser.add_argument('--output_dir', type=str, default='dataset')
    parser.add_argument('--data_root', type=str, default='../../datasets')

    args = parser.parse_args()

    if args.dataset == 'all':
        datasets = ['ACM', 'DBLP', 'IMDB']
    else:
        datasets = [args.dataset]

    for dataset_name in datasets:
        try:
            convert_hete_to_homo(dataset_name, args.output_dir, args.data_root)
        except Exception as e:
            print(f"\n[ERROR] {dataset_name} 转换失败: {e}\n")

    print(f"\n{'='*60}")
    print("转换完成!")
    print(f"{'='*60}\n")
