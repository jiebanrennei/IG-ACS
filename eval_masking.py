"""
遮蔽恢复评估指标模块 (Masking Recovery Evaluation Metrics)

新增指标：
- ARR (Adversarial Recovery Rate): 对抗恢复率
- MDR (Masking Detection Rate): 遮蔽检测率
- ACS (Adversarial Community Score): 对抗社区得分
- AMR (Anti-Masking Robustness): 抗遮蔽鲁棒性
- Intent Alignment: 意图对齐度
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional


def adversarial_recovery_rate(pred_community: Set[int],
                               true_community: Set[int],
                               masked_nodes: Set[int]) -> float:
    """
    对抗恢复率 (ARR)

    在所有被遮蔽（边被删）的社区成员中，方法成功找回了多少？

    ARR = |{v ∈ C* ∩ C' : v ∈ masked_nodes}| / |{v ∈ C* : v ∈ masked_nodes}|

    Args:
        pred_community: 预测的社区节点集合
        true_community: 真实的社区节点集合
        masked_nodes: 被遮蔽的节点集合（边被删的节点）

    Returns:
        ARR 值，范围 [0, 1]
    """
    if len(masked_nodes) == 0:
        return 0.0

    # 被隐藏的真实成员
    hidden_true = true_community & masked_nodes

    if len(hidden_true) == 0:
        return 0.0

    # 被找回的隐藏成员
    recovered = pred_community & hidden_true

    return len(recovered) / len(hidden_true)


def masking_detection_rate(pred_edges: Set[Tuple[int, int]],
                           deleted_edges: List[Tuple[int, int]],
                           added_edges: List[Tuple[int, int]]) -> float:
    """
    遮蔽检测率 (MDR)

    方法能正确识别出多少被篡改的边？

    MDR = (TP_del + TP_add) / (|E⁻| + |E⁺|)

    Args:
        pred_edges: 预测为"被篡改"的边集合
        deleted_edges: 实际被删除的边列表
        added_edges: 实际被添加的边列表

    Returns:
        MDR 值，范围 [0, 1]
    """
    total_tampered = len(deleted_edges) + len(added_edges)

    if total_tampered == 0:
        return 0.0

    # 正确识别的被删边
    tp_del = len(pred_edges & set(deleted_edges))

    # 正确识别的被加边
    tp_add = len(pred_edges & set(added_edges))

    return (tp_del + tp_add) / total_tampered


def intent_alignment(community_embedding: np.ndarray,
                     intent_vector: np.ndarray) -> float:
    """
    意图对齐度 (Intent Alignment)

    社区表示与查询意图的余弦相似度

    Args:
        community_embedding: 社区表示向量 (平均节点表示)
        intent_vector: 意图向量

    Returns:
        余弦相似度，范围 [-1, 1]
    """
    # 归一化
    norm_comm = np.linalg.norm(community_embedding)
    norm_intent = np.linalg.norm(intent_vector)

    if norm_comm < 1e-8 or norm_intent < 1e-8:
        return 0.0

    cos_sim = np.dot(community_embedding, intent_vector) / (norm_comm * norm_intent)

    return float(cos_sim)


def adversarial_community_score(f1: float,
                                 arr: float,
                                 intent_align: float,
                                 alpha: float = 0.4,
                                 beta: float = 0.4,
                                 gamma: float = 0.2) -> float:
    """
    对抗社区得分 (ACS)

    综合衡量恢复的社区的"对抗性质量"

    ACS = α·F1 + β·ARR + γ·IntentAlign

    Args:
        f1: 社区搜索 F1 值
        arr: 对抗恢复率
        intent_align: 意图对齐度
        alpha: F1 权重
        beta: ARR 权重
        gamma: 意图对齐权重

    Returns:
        ACS 值
    """
    return alpha * f1 + beta * arr + gamma * intent_align


def anti_masking_robustness(f1_scores: List[float]) -> float:
    """
    抗遮蔽鲁棒性 (AMR)

    在不同遮蔽强度下，性能下降的缓急程度

    AMR = mean(F1(ρ_k) / F1(ρ_0))

    Args:
        f1_scores: 不同遮蔽强度下的 F1 列表
                   f1_scores[0] 是原始图的 F1
                   f1_scores[1:] 是不同遮蔽强度下的 F1

    Returns:
        AMR 值，越接近 1 越鲁棒
    """
    if len(f1_scores) == 0 or f1_scores[0] < 1e-8:
        return 0.0

    f1_base = f1_scores[0]
    ratios = [f1 / f1_base for f1 in f1_scores]

    return float(np.mean(ratios))


def compute_all_masking_metrics(pred_community: Set[int],
                                 true_community: Set[int],
                                 pred_edges: Set[Tuple[int, int]],
                                 deleted_edges: List[Tuple[int, int]],
                                 added_edges: List[Tuple[int, int]],
                                 node_embeddings: np.ndarray,
                                 intent_vector: np.ndarray,
                                 f1_score: float) -> Dict[str, float]:
    """
    计算所有遮蔽恢复评估指标

    Args:
        pred_community: 预测的社区节点集合
        true_community: 真实的社区节点集合
        pred_edges: 预测为被篡改的边集合
        deleted_edges: 实际被删除的边列表
        added_edges: 实际被添加的边列表
        node_embeddings: 节点嵌入矩阵 (num_nodes, dim)
        intent_vector: 意图向量
        f1_score: 社区搜索 F1 值

    Returns:
        包含所有指标的字典
    """
    # 被遮蔽的节点
    masked_nodes = set()
    for u, v in deleted_edges:
        masked_nodes.add(u)
        masked_nodes.add(v)

    # ARR: 对抗恢复率
    arr = adversarial_recovery_rate(pred_community, true_community, masked_nodes)

    # MDR: 遮蔽检测率
    mdr = masking_detection_rate(pred_edges, deleted_edges, added_edges)

    # 意图对齐度
    if len(pred_community) > 0:
        community_indices = list(pred_community)
        community_embedding = node_embeddings[community_indices].mean(axis=0)
        intent_align = intent_alignment(community_embedding, intent_vector)
    else:
        intent_align = 0.0

    # ACS: 对抗社区得分
    acs = adversarial_community_score(f1_score, arr, intent_align)

    return {
        'F1': f1_score,
        'ARR': arr,
        'MDR': mdr,
        'IntentAlign': intent_align,
        'ACS': acs,
    }


def print_metrics_table(metrics_dict: Dict[str, Dict[str, float]],
                        title: str = "Evaluation Metrics"):
    """
    打印指标对比表格

    Args:
        metrics_dict: {方法名: {指标名: 值}}
        title: 表格标题
    """
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    # 获取所有指标名
    all_metrics = set()
    for metrics in metrics_dict.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)

    # 表头
    header = f"{'Method':<15}"
    for metric in all_metrics:
        header += f" | {metric:>10}"
    print(header)
    print("-" * len(header))

    # 数据行
    for method, metrics in metrics_dict.items():
        row = f"{method:<15}"
        for metric in all_metrics:
            value = metrics.get(metric, 0.0)
            row += f" | {value:>10.4f}"
        print(row)

    print(f"{'='*70}\n")


def compute_robustness_curve(masker,
                              method_name: str,
                              rho_values: List[float],
                              eval_func) -> Dict[str, List[float]]:
    """
    计算鲁棒性曲线：不同遮蔽强度下的性能

    Args:
        masker: HeterogeneousGraphMasker 实例
        method_name: 遮蔽方法名称 ('random', 'dice', 'marginal')
        rho_values: 遮蔽强度列表，如 [0.1, 0.2, 0.3, 0.4, 0.5]
        eval_func: 评估函数，输入 (adj_masked, edges_del, edges_add) 返回 F1

    Returns:
        {'rho_values': [...], 'f1_scores': [...], 'amr': ...}
    """
    f1_scores = []

    for rho in rho_values:
        if method_name == 'random':
            adj_masked, edges_del, edges_add = masker.random_mask(
                rho_delete=rho, rho_add=rho * 0.3
            )
        elif method_name == 'dice':
            S_plus = set(np.where(masker.labels == 0)[0])
            adj_masked, edges_del, edges_add = masker.dice_mask(
                S_plus, budget=rho
            )
        elif method_name == 'marginal':
            adj_masked, edges_del, edges_add, _ = masker.marginal_node_mask(
                budget=rho
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")

        f1 = eval_func(adj_masked, edges_del, edges_add)
        f1_scores.append(f1)

    amr = anti_masking_robustness(f1_scores)

    return {
        'rho_values': rho_values,
        'f1_scores': f1_scores,
        'amr': amr,
    }


if __name__ == '__main__':
    # 测试代码
    print("Testing evaluation metrics...")

    # 模拟数据
    pred_community = {1, 2, 3, 4, 5}
    true_community = {1, 2, 3, 6, 7}
    masked_nodes = {1, 2, 6}

    arr = adversarial_recovery_rate(pred_community, true_community, masked_nodes)
    print(f"ARR: {arr:.4f}")  # 应该 = 2/3 (节点 1, 2 被找回)

    pred_edges = {(1, 2), (3, 6)}
    deleted_edges = [(1, 2), (4, 5)]
    added_edges = [(3, 6), (7, 8)]

    mdr = masking_detection_rate(pred_edges, deleted_edges, added_edges)
    print(f"MDR: {mdr:.4f}")  # 应该 = 2/4 = 0.5

    f1 = 0.75
    intent_align = 0.8
    acs = adversarial_community_score(f1, arr, intent_align)
    print(f"ACS: {acs:.4f}")

    f1_scores = [0.8, 0.75, 0.7, 0.6, 0.5]
    amr = anti_masking_robustness(f1_scores)
    print(f"AMR: {amr:.4f}")
