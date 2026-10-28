"""
遮蔽恢复实验主流程 (Masking Recovery Experiment Runner)

运行所有方法在遮蔽数据集上的测试，计算所有指标，生成对比表格。

Usage:
    python run_masking_experiment.py --dataset ACM --mask_type all
    python run_masking_experiment.py --dataset all --mask_type marginal
"""

import os
import sys
import json
import argparse
import numpy as np
import scipy.sparse as sp
from typing import Dict, List
from collections import defaultdict

from community_masking import HeterogeneousGraphMasker, generate_all_masked_datasets
from eval_masking import (
    compute_all_masking_metrics,
    print_metrics_table,
    adversarial_recovery_rate,
    masking_detection_rate,
    adversarial_community_score,
    anti_masking_robustness,
)


def load_masked_data(mask_dir: str):
    """加载遮蔽后的数据"""
    adj_masked = sp.load_npz(os.path.join(mask_dir, 'meta_adj.npz'))
    edges_deleted = np.load(os.path.join(mask_dir, 'edges_deleted.npy'),
                           allow_pickle=True).tolist()
    edges_added = np.load(os.path.join(mask_dir, 'edges_added.npy'),
                         allow_pickle=True).tolist()

    with open(os.path.join(mask_dir, 'mask_info.json'), 'r') as f:
        info = json.load(f)

    return adj_masked, edges_deleted, edges_added, info


class MaskingExperimentRunner:
    """遮蔽恢复实验运行器"""

    def __init__(self, dataset_name: str, data_dir: str = './datasets',
                 masked_data_dir: str = './data'):
        self.dataset_name = dataset_name.upper()
        self.data_dir = data_dir
        self.masked_data_dir = masked_data_dir

        # 加载原始数据
        self.masker = HeterogeneousGraphMasker(dataset_name, data_dir)

        # 结果存储
        self.results = defaultdict(dict)

    def run_single_method(self, mask_type: str, method_name: str = 'IG-HII'):
        """
        在单个遮蔽方法上运行所有评估

        Args:
            mask_type: 遮蔽类型 ('random', 'degree', 'dice', 'marginal')
            method_name: 方法名称
        """
        mask_dir = f'{self.masked_data_dir}/{self.dataset_name}_masked/{mask_type}'

        if not os.path.exists(mask_dir):
            print(f"Warning: {mask_dir} not found, skipping...")
            return

        print(f"\nRunning evaluation on {self.dataset_name} - {mask_type}...")

        # 加载遮蔽数据
        adj_masked, edges_deleted, edges_added, info = load_masked_data(mask_dir)

        # 这里需要根据你的模型实现实际的评估
        # 以下为模拟结果
        print(f"  Original edges: {info['num_edges_original']}")
        print(f"  Masked edges: {info['num_edges_masked']}")
        print(f"  Deleted edges: {len(edges_deleted)}")
        print(f"  Added edges: {len(edges_added)}")

        # TODO: 这里需要调用你的模型进行实际的社区搜索
        # 以下为示例代码
        pred_community, pred_edges, node_embeddings, intent_vector = \
            self._run_community_search(adj_masked)

        # 计算真实社区（基于标签）
        true_community = set(np.where(self.masker.labels == 0)[0])

        # 计算 F1（需要根据你的评估代码）
        f1_score = self._compute_f1(pred_community, true_community)

        # 计算所有指标
        metrics = compute_all_masking_metrics(
            pred_community=pred_community,
            true_community=true_community,
            pred_edges=pred_edges,
            deleted_edges=edges_deleted,
            added_edges=edges_added,
            node_embeddings=node_embeddings,
            intent_vector=intent_vector,
            f1_score=f1_score,
        )

        self.results[mask_type][method_name] = metrics

        print(f"  F1: {metrics['F1']:.4f}")
        print(f"  ARR: {metrics['ARR']:.4f}")
        print(f"  MDR: {metrics['MDR']:.4f}")
        print(f"  ACS: {metrics['ACS']:.4f}")

        return metrics

    def _run_community_search(self, adj_masked):
        """
        运行社区搜索（需要替换为你的模型）

        TODO: 这里需要调用 train_ig.py 中的模型

        Returns:
            pred_community: 预测的社区节点集合
            pred_edges: 预测的被篡改边集合
            node_embeddings: 节点嵌入矩阵
            intent_vector: 意图向量
        """
        # 模拟结果
        num_nodes = adj_masked.shape[0]

        # 随机预测（需要替换为实际模型）
        pred_community = set(np.random.choice(num_nodes, size=100, replace=False))
        pred_edges = set()  # 需要模型预测

        # 模拟节点嵌入
        node_embeddings = np.random.randn(num_nodes, 256).astype(np.float32)
        intent_vector = np.random.randn(256).astype(np.float32)

        return pred_community, pred_edges, node_embeddings, intent_vector

    def _compute_f1(self, pred_community, true_community):
        """
        计算 F1 值（需要替换为你的评估代码）

        TODO: 这里需要调用 eval.py 中的评估函数
        """
        if len(pred_community) == 0 or len(true_community) == 0:
            return 0.0

        intersection = len(pred_community & true_community)
        precision = intersection / len(pred_community) if len(pred_community) > 0 else 0
        recall = intersection / len(true_community) if len(true_community) > 0 else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def run_all_methods(self, method_name: str = 'IG-HII'):
        """运行所有遮蔽方法"""
        mask_types = ['random', 'degree', 'dice', 'marginal']

        for mask_type in mask_types:
            self.run_single_method(mask_type, method_name)

        return self.results

    def print_results(self):
        """打印结果表格"""
        for mask_type, methods in self.results.items():
            print_metrics_table(
                methods,
                title=f"{self.dataset_name} - {mask_type} Masking"
            )

    def save_results(self, output_file: str = None):
        """保存结果到文件"""
        if output_file is None:
            output_file = f'results/{self.dataset_name}_masking_results.json'

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(dict(self.results), f, indent=2)

        print(f"Results saved to: {output_file}")


def run_robustness_experiment(dataset_name: str,
                               mask_type: str = 'random',
                               rho_values: List[float] = None):
    """
    运行鲁棒性实验：测试不同遮蔽强度下的性能

    Args:
        dataset_name: 数据集名称
        mask_type: 遮蔽方法
        rho_values: 遮蔽强度列表
    """
    if rho_values is None:
        rho_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    masker = HeterogeneousGraphMasker(dataset_name)

    print(f"\nRunning robustness experiment on {dataset_name} - {mask_type}")
    print(f"Rho values: {rho_values}")

    f1_scores = []

    for rho in rho_values:
        if mask_type == 'random':
            adj_masked, edges_del, edges_add = masker.random_mask(
                rho_delete=rho, rho_add=rho * 0.3
            )
        elif mask_type == 'dice':
            S_plus = set(np.where(masker.labels == 0)[0])
            adj_masked, edges_del, edges_add = masker.dice_mask(
                S_plus, budget=rho
            )
        elif mask_type == 'marginal':
            adj_masked, edges_del, edges_add, _ = masker.marginal_node_mask(
                budget=rho
            )
        else:
            raise ValueError(f"Unknown method: {mask_type}")

        # TODO: 这里需要调用实际模型计算 F1
        # 以下为模拟
        f1 = 0.7 - rho * 0.3  # 模拟性能下降
        f1_scores.append(f1)

        print(f"  rho={rho:.1f}: F1={f1:.4f}")

    amr = anti_masking_robustness(f1_scores)
    print(f"\n  AMR (Anti-Masking Robustness): {amr:.4f}")

    return {
        'rho_values': rho_values,
        'f1_scores': f1_scores,
        'amr': amr,
    }


def main():
    parser = argparse.ArgumentParser(description='Run masking recovery experiments')
    parser.add_argument('--dataset', type=str, default='ACM',
                       choices=['ACM', 'DBLP', 'IMDB', 'all'],
                       help='Dataset name')
    parser.add_argument('--mask_type', type=str, default='all',
                       choices=['random', 'degree', 'dice', 'marginal', 'all'],
                       help='Masking type')
    parser.add_argument('--generate_data', action='store_true',
                       help='Generate masked datasets first')
    parser.add_argument('--robustness', action='store_true',
                       help='Run robustness experiment')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory')

    args = parser.parse_args()

    # 生成遮蔽数据
    if args.generate_data:
        print("Generating masked datasets...")
        generate_all_masked_datasets(args.data_dir, args.output_dir)

    # 运行实验
    datasets = ['ACM', 'DBLP', 'IMDB'] if args.dataset == 'all' else [args.dataset]
    mask_types = ['random', 'degree', 'dice', 'marginal'] if args.mask_type == 'all' else [args.mask_type]

    for dataset in datasets:
        runner = MaskingExperimentRunner(dataset, args.data_dir, args.output_dir)

        if args.robustness:
            # 鲁棒性实验
            for mask_type in mask_types:
                run_robustness_experiment(dataset, mask_type)
        else:
            # 正常实验
            for mask_type in mask_types:
                runner.run_single_method(mask_type, method_name='IG-HII')

            runner.print_results()
            runner.save_results()


if __name__ == '__main__':
    main()
