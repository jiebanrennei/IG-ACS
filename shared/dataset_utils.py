"""
通用数据集管理模块
所有基线方法共享，自动检测支持的数据集
"""

import os
import numpy as np
from pathlib import Path


# 支持的异构图数据集
HETE_DATASETS = {
    'ACM': {
        'dir': 'acm',
        'labels': 'labels.npy',
        'node_types': ['paper', 'author', 'subject'],
        'features': {
            'paper': 'p_feat.npz',
            'author': 'a_feat.npz'
        },
        'meta_paths': ['pap.npz', 'psp.npz']
    },
    'DBLP': {
        'dir': 'dblp',
        'labels': 'labels.npy',
        'node_types': ['author', 'paper', 'conference', 'term'],
        'features': {
            'author': 'a_feat.npz',
            'paper': 'p_feat.npz',
            'term': 't_feat.npz'
        },
        'meta_paths': ['apa.npz', 'apcpa.npz', 'aptpa.npz']
    },
    'IMDB': {
        'dir': 'imdb_new',
        'labels': 'labels.npy',
        'node_types': ['movie', 'actor', 'director'],
        'features': {
            'movie': 'm_feat.npz'
        },
        'meta_paths': ['mam.npz', 'mdm.npz']
    }
}

# 支持的同构图数据集
HOMO_DATASETS = [
    'cora', 'citeseer', 'pubmed',
    'amazon', 'dblp', 'lj', 'youtube', 'twitter', 'facebook',
    'com-amazon', 'com-dblp', 'com-lj', 'com-youtube', 'com-twitter'
]


def get_dataset_list(data_root='./datasets'):
    """获取数据根目录下所有可用的数据集"""
    datasets = []

    if not os.path.exists(data_root):
        return datasets

    for name in os.listdir(data_root):
        path = os.path.join(data_root, name)
        if os.path.isdir(path):
            # 检查是否是有效的数据集目录
            if os.path.exists(os.path.join(path, 'labels.npy')):
                datasets.append(name.upper())

    return sorted(datasets)


def is_hete_dataset(dataset_name):
    """判断是否是异构图数据集"""
    return dataset_name.upper() in HETE_DATASETS


def is_homo_dataset(dataset_name):
    """判断是否是同构图数据集"""
    return dataset_name.lower() in HOMO_DATASETS


def get_dataset_info(dataset_name, data_root='./datasets'):
    """获取数据集信息"""
    dataset_name = dataset_name.upper()

    if dataset_name in HETE_DATASETS:
        cfg = HETE_DATASETS[dataset_name]
        data_dir = os.path.join(data_root, cfg['dir'])

        # 加载标签
        labels = np.load(os.path.join(data_dir, cfg['labels']))

        return {
            'name': dataset_name,
            'type': 'heterogeneous',
            'dir': data_dir,
            'labels': labels,
            'num_classes': len(np.unique(labels)),
            'config': cfg
        }

    else:
        # 同构图数据集
        data_dir = os.path.join(data_root, dataset_name.lower())
        if not os.path.exists(data_dir):
            raise ValueError(f"Dataset {dataset_name} not found in {data_root}")

        return {
            'name': dataset_name,
            'type': 'homogeneous',
            'dir': data_dir
        }


def load_pyg_data(dataset_name, data_root='./dataset'):
    """加载 PyG 格式的数据集（用于 EDA-GCL 等）"""
    import torch

    dataset_name = dataset_name.upper()

    # 尝试加载转换后的 PyG 数据
    pyg_path = os.path.join(data_root, dataset_name.lower(), f'{dataset_name.lower()}_pyg.pt')

    if os.path.exists(pyg_path):
        data = torch.load(pyg_path)
        return data
    else:
        raise FileNotFoundError(f"PyG data not found: {pyg_path}. Please run conversion first.")


def print_dataset_info(data_root='./datasets'):
    """打印所有可用数据集的信息"""
    datasets = get_dataset_list(data_root)

    print(f"\n{'='*60}")
    print(f"可用数据集 (共 {len(datasets)} 个)")
    print(f"{'='*60}")

    for name in datasets:
        try:
            info = get_dataset_info(name, data_root)
            dataset_type = "异构图" if info['type'] == 'heterogeneous' else "同构图"

            if info['type'] == 'heterogeneous':
                print(f"\n{name} ({dataset_type})")
                print(f"  类别数: {info['num_classes']}")
                print(f"  节点类型: {', '.join(info['config']['node_types'])}")
            else:
                print(f"\n{name} ({dataset_type})")

        except Exception as e:
            print(f"\n{name}: 加载失败 ({e})")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    # 打印所有可用数据集
    print_dataset_info()
