import torch
import os
import sys
import numpy as np
import random

from torch_geometric.datasets import (
    Planetoid, CitationFull, Amazon, Coauthor,
    WikipediaNetwork, WebKB, Actor
)
import torch_geometric.transforms as T
from deeprobust.graph.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# 添加共享模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.dataset_utils import HETE_DATASETS


# =============================================================================
# Random Seed Configuration
# =============================================================================

def set_everything(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Dataset Loading Utilities
# =============================================================================

def get_dataset(path, name):
    # 支持的数据集列表（动态生成）
    supported_datasets = [
        'Cora', 'CiteSeer', "AmazonC", "AmazonP",
        'CoauthorC', 'CoauthorP', "PubMed",
        'cora_lcc', 'citeseer_lcc',
        'Cornell', 'Texas', 'Wisconsin',
        'chameleon', 'squirrel', 'Actor',
    ]
    # 添加所有异构图数据集
    supported_datasets.extend(list(HETE_DATASETS.keys()))

    # Validate dataset name
    assert name in supported_datasets, f"Unknown dataset: {name}. Supported: {supported_datasets}"

    # -------------------------------------------------------------------------
    # Heterogeneous Graph Datasets (converted to homogeneous)
    # -------------------------------------------------------------------------
    if name in HETE_DATASETS:
        # 加载转换后的 PyG 数据 (保存在 ./dataset/ 目录)
        data_path = f'./dataset/{name.lower()}/{name.lower()}_pyg.pt'
        data = torch.load(data_path)
        print(f"Loaded {name} dataset: {data.num_nodes} nodes, {data.num_edges} edges")
        return [data]
    
    # -------------------------------------------------------------------------
    # Heterophilous Graph Datasets
    # -------------------------------------------------------------------------
    if name == "Actor":
        path = f'{path}/{name}'
        return Actor(path, transform=T.NormalizeFeatures())
    
    if name in ['Cornell', 'Texas', 'Wisconsin']:
        return WebKB(path, name, transform=T.NormalizeFeatures())
    
    if name in ['chameleon', 'squirrel']:
        return WikipediaNetwork(path, name, transform=T.NormalizeFeatures())
    
    # -------------------------------------------------------------------------
    # Amazon Datasets
    # -------------------------------------------------------------------------
    if name == "AmazonC":
        return Amazon(path, "Computers", T.NormalizeFeatures())
    
    if name == "AmazonP":
        return Amazon(path, "Photo", T.NormalizeFeatures())
    
    # -------------------------------------------------------------------------
    # Coauthor Datasets
    # -------------------------------------------------------------------------
    if name == 'CoauthorC':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    
    if name == 'CoauthorP':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())
    
    # -------------------------------------------------------------------------
    # DeepRobust Datasets (LCC variants)
    # -------------------------------------------------------------------------
    if name == "cora_lcc":
        name = "cora"
        data = Dataset(root=path, name=name, setting='prognn')
        adj, features, labels = data.adj, data.features, data.labels
        dataset = Data()
        dataset.x = torch.from_numpy(features.toarray()).float()
        dataset.y = torch.from_numpy(labels).long()
        dataset.edge_index = dense_to_sparse(torch.from_numpy(adj.toarray()))[0].long()
        return [dataset]
    
    if name == "citeseer_lcc":
        name = "citeseer"
        data = Dataset(root=path, name=name, setting='prognn')
        adj, features, labels = data.adj, data.features, data.labels
        dataset = Data()
        dataset.x = torch.from_numpy(features.toarray()).float()
        dataset.y = torch.from_numpy(labels).long()
        dataset.edge_index = dense_to_sparse(torch.from_numpy(adj.toarray()))[0].long()
        return [dataset]
    
    # -------------------------------------------------------------------------
    # Default: Planetoid Datasets (Cora, CiteSeer, PubMed)
    # -------------------------------------------------------------------------
    return Planetoid(
        path,
        name,
        "public",
        T.NormalizeFeatures()
    )