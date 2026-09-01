import torch
import os
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
    # Validate dataset name
    assert name in [
        'Cora', 'CiteSeer', "AmazonC", "AmazonP",
        'CoauthorC', 'CoauthorP', "PubMed",
        'cora_lcc', 'citeseer_lcc',
        'Cornell', 'Texas', 'Wisconsin',
        'chameleon', 'squirrel', 'Actor',
        'ACM', 'DBLP', 'IMDB'  # 新增异构图数据集
    ]

    # -------------------------------------------------------------------------
    # Heterogeneous Graph Datasets (converted to homogeneous)
    # -------------------------------------------------------------------------
    if name in ['ACM', 'DBLP', 'IMDB']:
        # 加载转换后的 PyG 数据
        data_path = f'{path}/../dataset/{name.lower()}/{name.lower()}_pyg.pt'
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