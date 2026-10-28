"""
社区遮蔽模块 (Community Masking Module)

用于在异构图数据集上生成对抗性遮蔽数据，验证 IG-ACS 的社区恢复能力。

遮蔽方法：
1. 随机遮蔽 (Random Masking) - baseline
2. 度中心性遮蔽 (Degree-based Masking) - baseline
3. DICE 攻击 (DICE Attack) - 借鉴 Chen et al. 基线
4. 边缘节点遮蔽 (Marginal Node Masking) - 核心方法，借鉴 Chen et al. TCSS 2024

参考文献：
- Chen C, et al. Privacy Protection for Marginal-Sensitive Community Individuals
  Against Adversarial Community Detection Attacks. IEEE TCSS, 2024.
"""

import os
import json
import random
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from collections import defaultdict

import networkx as nx
from networkx.algorithms import community as nx_community

# 尝试导入 louvain
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("Warning: python-louvain not installed. Using alternative community detection.")


class HeterogeneousGraphMasker:
    """
    异构图遮蔽器

    在元路径邻接矩阵上进行遮蔽，支持 ACM/DBLP/IMDB 等异构图数据集。
    """

    def __init__(self, dataset_name: str, data_dir: str = './datasets'):
        """
        Args:
            dataset_name: 数据集名称 ('ACM', 'DBLP', 'IMDB')
            data_dir: 数据集根目录
        """
        self.dataset_name = dataset_name.upper()
        self.data_dir = data_dir

        # 数据集配置
        self.dataset_configs = {
            'ACM': {
                'dir': 'acm',
                'feat': 'p_feat.npz',
                'meta_paths': ['pap.npz', 'psp.npz'],
                'default_mp': 'pap.npz',
                'labels_file': 'labels.npy',
            },
            'DBLP': {
                'dir': 'dblp',
                'feat': 'a_feat.npz',
                'meta_paths': ['apa.npz', 'apcpa.npz', 'aptpa.npz'],
                'default_mp': 'apa.npz',
                'labels_file': 'labels.npy',
            },
            'IMDB': {
                'dir': 'self_imdb',
                'feat': 'm_feat.npz',
                'meta_paths': ['mam.npz', 'mdm.npz'],
                'default_mp': 'mam.npz',
                'labels_file': 'labels.npy',
            },
        }

        if self.dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(self.dataset_configs.keys())}")

        self.config = self.dataset_configs[self.dataset_name]
        self.base_path = os.path.join(data_dir, self.config['dir'])

        # 加载数据
        self._load_data()

    def _load_data(self):
        """加载元路径邻接矩阵和标签"""
        # 加载默认元路径邻接矩阵
        mp_file = os.path.join(self.base_path, self.config['default_mp'])
        self.adj = sp.load_npz(mp_file)
        self.adj = (self.adj > 0).astype(np.uint8)

        # 确保对称
        self.adj = self.adj.maximum(self.adj.T)

        # 加载标签
        labels_file = os.path.join(self.base_path, self.config['labels_file'])
        self.labels = np.load(labels_file)

        self.num_nodes = self.adj.shape[0]
        self.num_classes = len(np.unique(self.labels))

        print(f"Loaded {self.dataset_name}: {self.num_nodes} nodes, "
              f"{self.adj.nnz // 2} edges, {self.num_classes} classes")

    # =========================================================================
    # 边缘节点选择 (借鉴 Chen et al. TCSS 2024)
    # =========================================================================

    def select_marginal_nodes(self,
                              methods: List[str] = ['louvain', 'label_propagation'],
                              distance_threshold: int = 2) -> Set[int]:
        """
        用多种社区检测算法找边缘节点，取交集

        Args:
            methods: 社区检测算法列表 ['louvain', 'label_propagation', 'async_lpa']
            distance_threshold: 边缘节点距离阈值

        Returns:
            边缘节点集合 S+
        """
        G = self._to_networkx()

        marginal_sets = []

        for method in methods:
            if method == 'louvain':
                if HAS_LOUVAIN:
                    partition = community_louvain.best_partition(G, random_state=42)
                    # 转换为社区列表
                    comm_dict = defaultdict(set)
                    for node, comm_id in partition.items():
                        comm_dict[comm_id].add(node)
                    communities = list(comm_dict.values())
                else:
                    # 备选：用标签传播
                    communities = list(nx_community.label_propagation_communities(G))
                    print("  [louvain] python-louvain not available, using label_propagation instead")
            elif method == 'gn':
                # GN 算法太慢，跳过
                print("  [gn] Skipping GN (too slow for large graphs)")
                continue
            elif method == 'label_propagation' or method == 'async_lpa':
                communities = list(nx_community.label_propagation_communities(G))
            elif method == 'greedy_modularity':
                communities = list(nx_community.greedy_modularity_communities(G))
            else:
                raise ValueError(f"Unknown method: {method}")

            marginal = self._get_marginal_nodes(communities, G, distance_threshold)
            marginal_sets.append(marginal)
            print(f"  [{method}] Found {len(marginal)} marginal nodes")

        # 取交集
        if len(marginal_sets) == 0:
            return set()

        S_plus = marginal_sets[0]
        for s in marginal_sets[1:]:
            S_plus = S_plus & s

        print(f"  Intersection: {len(S_plus)} marginal nodes")
        return S_plus

    def _get_marginal_nodes(self,
                            communities: List[Set[int]],
                            G: nx.Graph,
                            distance_threshold: int = 2) -> Set[int]:
        """
        获取边缘节点：与相邻社区有连接的节点

        Args:
            communities: 社区列表
            G: NetworkX 图
            distance_threshold: 距离阈值

        Returns:
            边缘节点集合
        """
        marginal = set()

        # 为每个节点标记所属社区
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = i

        # 找边缘节点：有边连接到其他社区的节点
        for node in G.nodes():
            if node not in node_to_comm:
                continue

            node_comm = node_to_comm[node]

            for neighbor in G.neighbors(node):
                if neighbor in node_to_comm:
                    neighbor_comm = node_to_comm[neighbor]
                    if neighbor_comm != node_comm:
                        # 有边连接到其他社区，是边缘节点
                        marginal.add(node)
                        break

        return marginal

    def _to_networkx(self) -> nx.Graph:
        """将稀疏邻接矩阵转为 NetworkX 图"""
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))

        # 添加边
        rows, cols = self.adj.nonzero()
        for i, j in zip(rows, cols):
            if i < j:  # 只加上三角，避免重复
                G.add_edge(i, j)

        return G

    # =========================================================================
    # 遮蔽方法
    # =========================================================================

    def random_mask(self,
                    S_plus: Optional[Set[int]] = None,
                    rho_delete: float = 0.3,
                    rho_add: float = 0.1,
                    seed: int = 42) -> Tuple[sp.csr_matrix, List, List]:
        """
        随机遮蔽：在元路径矩阵上随机删边/加边

        Args:
            S_plus: 目标节点集合，None 表示全图
            rho_delete: 删边比例
            rho_add: 加边比例
            seed: 随机种子

        Returns:
            (遮蔽后的邻接矩阵, 删除的边列表, 添加的边列表)
        """
        random.seed(seed)
        np.random.seed(seed)

        # 转为 COO 格式方便操作
        adj_coo = self.adj.tocoo()
        rows, cols = adj_coo.row, adj_coo.col

        # 只保留上三角（无向图）
        mask = rows < cols
        rows, cols = rows[mask], cols[mask]

        if S_plus is None:
            S_plus_set = set(range(self.num_nodes))
            S_plus_array = np.arange(self.num_nodes)
        else:
            S_plus_set = S_plus
            S_plus_array = np.array(list(S_plus))

        # 找出 S+ 内部的边
        row_in_S = np.isin(rows, S_plus_array)
        col_in_S = np.isin(cols, S_plus_array)
        internal_mask = row_in_S & col_in_S

        internal_rows = rows[internal_mask]
        internal_cols = cols[internal_mask]

        # 随机选择要删除的边
        num_to_delete = int(len(internal_rows) * rho_delete)
        if num_to_delete > 0 and len(internal_rows) > 0:
            delete_idx = np.random.choice(len(internal_rows), size=num_to_delete, replace=False)
            delete_rows = internal_rows[delete_idx]
            delete_cols = internal_cols[delete_idx]
            edges_to_delete = list(zip(delete_rows.tolist(), delete_cols.tolist()))
        else:
            edges_to_delete = []
            delete_rows = np.array([], dtype=int)
            delete_cols = np.array([], dtype=int)

        # 创建删除掩码
        delete_set = set(zip(delete_rows.tolist(), delete_cols.tolist()))

        # 构建新的邻接矩阵
        keep_mask = ~np.array([(r, c) in delete_set for r, c in zip(rows, cols)])
        new_rows = rows[keep_mask]
        new_cols = cols[keep_mask]

        # 添加 S+ 与外部的边
        non_S_array = np.array([i for i in range(self.num_nodes) if i not in S_plus_set])
        edges_to_add = []

        if len(S_plus_array) > 0 and len(non_S_array) > 0:
            num_to_add = int(len(internal_rows) * rho_add)
            if num_to_add > 0:
                # 随机选择要添加的边
                for _ in range(min(num_to_add, len(S_plus_array) * len(non_S_array))):
                    u = random.choice(S_plus_array)
                    v = random.choice(non_S_array)
                    if self.adj[u, v] == 0:  # 确保原来没有边
                        edges_to_add.append((int(u), int(v)))

        # 构建最终的稀疏矩阵
        all_rows = list(new_rows) + [u for u, v in edges_to_add]
        all_cols = list(new_cols) + [v for u, v in edges_to_add]
        data = np.ones(len(all_rows))

        adj_masked = sp.coo_matrix((data, (all_rows, all_cols)), shape=self.adj.shape)
        adj_masked = adj_masked.maximum(adj_masked.T)  # 对称化
        adj_masked = adj_masked.tocsr()

        return adj_masked, edges_to_delete, edges_to_add

    def degree_mask(self,
                    S_plus: Optional[Set[int]] = None,
                    rho_delete: float = 0.3,
                    top_k_ratio: float = 0.2,
                    seed: int = 42) -> Tuple[sp.csr_matrix, List, List]:
        """
        度中心性遮蔽：优先删高度数节点的边

        Args:
            S_plus: 目标节点集合
            rho_delete: 删边比例
            top_k_ratio: 优先删除的高度数节点比例
            seed: 随机种子

        Returns:
            (遮蔽后的邻接矩阵, 删除的边列表, 添加的边列表)
        """
        random.seed(seed)
        np.random.seed(seed)

        adj_masked = self.adj.copy().tolil()

        if S_plus is None:
            S_plus = set(range(self.num_nodes))

        # 计算 S+ 内每个节点的度数
        degrees = {}
        for u in S_plus:
            degrees[u] = adj_masked[u].sum()

        # 按度数排序，优先删高度数节点的边
        sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
        top_k = int(len(sorted_nodes) * top_k_ratio)
        high_degree_nodes = set(sorted_nodes[:top_k])

        # 删除高度数节点的边
        edges_to_delete = []
        for u in high_degree_nodes:
            for v in S_plus:
                if u < v and adj_masked[u, v] > 0:
                    if random.random() < rho_delete:
                        edges_to_delete.append((u, v))

        for u, v in edges_to_delete:
            adj_masked[u, v] = 0
            adj_masked[v, u] = 0

        return adj_masked.tocsr(), edges_to_delete, []

    def dice_mask(self,
                  S_plus: Set[int],
                  budget: float = 0.2,
                  seed: int = 42) -> Tuple[sp.csr_matrix, List, List]:
        """
        DICE 攻击风格遮蔽：删 S+ 内部边 + 加 S+ 与外部的边

        借鉴 Chen et al. TCSS 2024 中使用的 DICE 基线方法

        Args:
            S_plus: 目标节点集合
            budget: 扰动预算（占总边数的比例）
            seed: 随机种子

        Returns:
            (遮蔽后的邻接矩阵, 删除的边列表, 添加的边列表)
        """
        random.seed(seed)
        np.random.seed(seed)

        adj_masked = self.adj.copy().tolil()

        # 计算预算
        internal_edges = [(u, v) for u in S_plus for v in S_plus
                         if u < v and adj_masked[u, v] > 0]

        num_to_delete = int(len(internal_edges) * budget / 2)
        num_to_add = int(len(internal_edges) * budget / 2)

        # 删除 S+ 内部边
        edges_to_delete = []
        if len(internal_edges) > 0:
            edges_to_delete = random.sample(
                internal_edges,
                min(num_to_delete, len(internal_edges))
            )

        for u, v in edges_to_delete:
            adj_masked[u, v] = 0
            adj_masked[v, u] = 0

        # 添加 S+ 与外部的边
        non_S = [i for i in range(self.num_nodes) if i not in S_plus]
        possible_add = [(u, v) for u in S_plus for v in non_S
                       if adj_masked[u, v] == 0]

        edges_to_add = []
        if len(possible_add) > 0:
            edges_to_add = random.sample(
                possible_add,
                min(num_to_add, len(possible_add))
            )

        for u, v in edges_to_add:
            adj_masked[u, v] = 1
            adj_masked[v, u] = 1

        return adj_masked.tocsr(), edges_to_delete, edges_to_add

    def marginal_node_mask(self,
                           budget: float = 0.3,
                           methods: List[str] = ['louvain', 'label_propagation'],
                           seed: int = 42) -> Tuple[sp.csr_matrix, List, List, Set[int]]:
        """
        边缘节点遮蔽：核心方法，借鉴 Chen et al. TCSS 2024

        1. 用多种社区检测算法找边缘节点 S+
        2. 删除 S+ 内部的边

        Args:
            budget: 删边预算（S+ 内部边的比例）
            methods: 社区检测算法列表
            seed: 随机种子

        Returns:
            (遮蔽后的邻接矩阵, 删除的边列表, 添加的边列表, S+节点集合)
        """
        random.seed(seed)
        np.random.seed(seed)

        # 选择边缘节点
        print("Selecting marginal nodes...")
        S_plus = self.select_marginal_nodes(methods=methods)

        if len(S_plus) == 0:
            print("Warning: No marginal nodes found, using random nodes")
            S_plus = set(random.sample(range(self.num_nodes), min(100, self.num_nodes)))

        adj_masked = self.adj.copy().tolil()

        # 删除 S+ 内部的边
        internal_edges = [(u, v) for u in S_plus for v in S_plus
                         if u < v and adj_masked[u, v] > 0]

        num_to_delete = int(len(internal_edges) * budget)
        edges_to_delete = []

        if len(internal_edges) > 0:
            edges_to_delete = random.sample(
                internal_edges,
                min(num_to_delete, len(internal_edges))
            )

        for u, v in edges_to_delete:
            adj_masked[u, v] = 0
            adj_masked[v, u] = 0

        return adj_masked.tocsr(), edges_to_delete, [], S_plus

    # =========================================================================
    # 数据保存
    # =========================================================================

    def save_masked_data(self,
                         adj_masked: sp.csr_matrix,
                         edges_deleted: List,
                         edges_added: List,
                         mask_name: str,
                         S_plus: Optional[Set[int]] = None,
                         output_dir: Optional[str] = None) -> str:
        """
        保存遮蔽后的数据

        Args:
            adj_masked: 遮蔽后的邻接矩阵
            edges_deleted: 删除的边列表
            edges_added: 添加的边列表
            mask_name: 遮蔽方法名称
            S_plus: 目标节点集合
            output_dir: 输出目录

        Returns:
            输出目录路径
        """
        if output_dir is None:
            output_dir = f'./data/{self.dataset_name}_masked/{mask_name}'

        os.makedirs(output_dir, exist_ok=True)

        # 保存遮蔽后的邻接矩阵
        sp.save_npz(os.path.join(output_dir, 'meta_adj.npz'), adj_masked)

        # 同时保存为元路径文件名（如 pap.npz），供 train_ig.py --perturbed_data 使用
        cfg = self.dataset_configs[self.dataset_name]
        default_mp = cfg['default_mp']
        sp.save_npz(os.path.join(output_dir, default_mp), adj_masked)

        # 复制原始特征、标签和所有元路径文件（供 train_ig.py 加载）
        import shutil
        src_dir = self.base_path
        files_to_copy = [cfg['feat'], cfg['labels_file']] + cfg['meta_paths']
        for fname in files_to_copy:
            src_file = os.path.join(src_dir, fname)
            dst_file = os.path.join(output_dir, fname)
            if os.path.exists(src_file) and not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)

        # 保存遮蔽信息
        info = {
            'dataset': self.dataset_name,
            'mask_name': mask_name,
            'num_nodes': int(self.num_nodes),
            'num_edges_original': int(self.adj.nnz // 2),
            'num_edges_masked': int(adj_masked.nnz // 2),
            'num_edges_deleted': int(len(edges_deleted)),
            'num_edges_added': int(len(edges_added)),
            'S_plus': [int(x) for x in S_plus] if S_plus is not None else None,
            'S_plus_size': int(len(S_plus)) if S_plus is not None else 0,
        }

        with open(os.path.join(output_dir, 'mask_info.json'), 'w') as f:
            json.dump(info, f, indent=2)

        # 保存边列表
        np.save(os.path.join(output_dir, 'edges_deleted.npy'),
                np.array([(int(u), int(v)) for u, v in edges_deleted]))
        np.save(os.path.join(output_dir, 'edges_added.npy'),
                np.array([(int(u), int(v)) for u, v in edges_added]))

        print(f"Saved masked data to: {output_dir}")
        return output_dir


def generate_single_masked_dataset(dataset_name: str,
                                    mask_type: str,
                                    data_dir: str = './datasets',
                                    output_base: str = './data',
                                    **kwargs):
    """
    生成单个遮蔽数据集

    Args:
        dataset_name: 数据集名称 ('ACM', 'DBLP', 'IMDB')
        mask_type: 遮蔽方法 ('random', 'degree', 'dice', 'marginal')
        data_dir: 原始数据目录
        output_base: 输出基础目录
        **kwargs: 遮蔽参数
    """
    print(f"\n{'='*60}")
    print(f"Generating {dataset_name} - {mask_type} masked dataset...")
    print(f"{'='*60}")

    masker = HeterogeneousGraphMasker(dataset_name, data_dir)

    if mask_type == 'random':
        rho_delete = kwargs.get('rho_delete', 0.3)
        rho_add = kwargs.get('rho_add', 0.1)
        adj_masked, edges_del, edges_add = masker.random_mask(
            rho_delete=rho_delete, rho_add=rho_add, seed=42
        )
        masker.save_masked_data(
            adj_masked, edges_del, edges_add,
            'random', output_dir=f'{output_base}/{dataset_name}_masked/random'
        )

    elif mask_type == 'degree':
        rho_delete = kwargs.get('rho_delete', 0.3)
        top_k_ratio = kwargs.get('top_k_ratio', 0.2)
        adj_masked, edges_del, edges_add = masker.degree_mask(
            rho_delete=rho_delete, top_k_ratio=top_k_ratio, seed=42
        )
        masker.save_masked_data(
            adj_masked, edges_del, edges_add,
            'degree', output_dir=f'{output_base}/{dataset_name}_masked/degree'
        )

    elif mask_type == 'dice':
        budget = kwargs.get('budget', 0.2)
        # 用标签找社区作为 S+
        S_plus = set(np.where(masker.labels == 0)[0])
        adj_masked, edges_del, edges_add = masker.dice_mask(
            S_plus, budget=budget, seed=42
        )
        masker.save_masked_data(
            adj_masked, edges_del, edges_add,
            'dice', S_plus=S_plus,
            output_dir=f'{output_base}/{dataset_name}_masked/dice'
        )

    elif mask_type == 'marginal':
        budget = kwargs.get('budget', 0.3)
        methods = kwargs.get('methods', ['louvain', 'label_propagation'])
        adj_masked, edges_del, edges_add, S_plus = masker.marginal_node_mask(
            budget=budget, methods=methods, seed=42
        )
        masker.save_masked_data(
            adj_masked, edges_del, edges_add,
            'marginal', S_plus=S_plus,
            output_dir=f'{output_base}/{dataset_name}_masked/marginal'
        )

    else:
        raise ValueError(f"Unknown mask_type: {mask_type}. "
                        f"Available: ['random', 'degree', 'dice', 'marginal']")

    print(f"\n{'='*60}")
    print(f"Done! Output: {output_base}/{dataset_name}_masked/{mask_type}/")
    print(f"{'='*60}")


def generate_all_masked_datasets(data_dir: str = './datasets',
                                  output_base: str = './data'):
    """
    为所有数据集生成所有遮蔽方法的测试数据

    Args:
        data_dir: 原始数据目录
        output_base: 输出基础目录
    """
    datasets = ['ACM', 'DBLP', 'IMDB']

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}...")
        print(f"{'='*60}")

        masker = HeterogeneousGraphMasker(dataset_name, data_dir)

        # 1. 随机遮蔽
        print("\n[1/4] Random masking...")
        adj_masked, edges_del, edges_add = masker.random_mask(
            rho_delete=0.3, rho_add=0.1, seed=42
        )
        masker.save_masked_data(
            adj_masked, edges_del, edges_add,
            'random', output_dir=f'{output_base}/{dataset_name}_masked/random'
        )

        # 2. 度中心性遮蔽
        print("\n[2/4] Degree-based masking...")
        adj_masked, edges_del, edges_add = masker.degree_mask(
            rho_delete=0.3, top_k_ratio=0.2, seed=42
        )
        masker.save_masked_data(
            adj_masked, edges_del, edges_add,
            'degree', output_dir=f'{output_base}/{dataset_name}_masked/degree'
        )

        # 3. DICE 遮蔽
        print("\n[3/4] DICE masking...")
        # 先用标签找社区作为 S+
        S_plus = set(np.where(masker.labels == 0)[0])  # 用类别 0 作为目标
        adj_masked, edges_del, edges_add = masker.dice_mask(
            S_plus, budget=0.2, seed=42
        )
        masker.save_masked_data(
            adj_masked, edges_del, edges_add,
            'dice', S_plus=S_plus,
            output_dir=f'{output_base}/{dataset_name}_masked/dice'
        )

        # 4. 边缘节点遮蔽
        print("\n[4/4] Marginal node masking...")
        adj_masked, edges_del, edges_add, S_plus = masker.marginal_node_mask(
            budget=0.3, methods=['louvain', 'label_propagation'], seed=42
        )
        masker.save_masked_data(
            adj_masked, edges_del, edges_add,
            'marginal', S_plus=S_plus,
            output_dir=f'{output_base}/{dataset_name}_masked/marginal'
        )

    print(f"\n{'='*60}")
    print("All masked datasets generated!")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate masked datasets for community search evaluation')
    parser.add_argument('--dataset', type=str, default='ACM',
                       choices=['ACM', 'DBLP', 'IMDB'],
                       help='Dataset name (default: ACM)')
    parser.add_argument('--mask_type', type=str, default='marginal',
                       choices=['random', 'degree', 'dice', 'marginal'],
                       help='Masking method (default: marginal)')
    parser.add_argument('--budget', type=float, default=0.3,
                       help='Perturbation budget for dice/marginal (default: 0.3)')
    parser.add_argument('--rho_delete', type=float, default=0.3,
                       help='Edge deletion ratio for random/degree (default: 0.3)')
    parser.add_argument('--rho_add', type=float, default=0.1,
                       help='Edge addition ratio for random (default: 0.1)')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                       help='Input data directory (default: ./datasets)')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory (default: ./data)')

    args = parser.parse_args()

    # 构建参数字典
    kwargs = {
        'budget': args.budget,
        'rho_delete': args.rho_delete,
        'rho_add': args.rho_add,
    }

    generate_single_masked_dataset(
        dataset_name=args.dataset,
        mask_type=args.mask_type,
        data_dir=args.data_dir,
        output_base=args.output_dir,
        **kwargs
    )
