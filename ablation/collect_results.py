#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验结果收集与对比
从日志文件中提取关键指标并生成对比表，按类别分组
"""

import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 消融实验描述和类别
ABLATION_INFO = {
    'full_model': {'desc': '完整模型（基线）', 'category': '基线'},
    # 损失函数消融
    'no_intent_loss': {'desc': '去除意图一致性损失', 'category': '损失函数'},
    'no_edge_feature_loss': {'desc': '去除边特征一致性损失', 'category': '损失函数'},
    # 模块级消融
    'no_rec_view': {'desc': '去除重构视图', 'category': '模块'},
    'no_edge_importance': {'desc': '去除边重要性', 'category': '模块'},
}

# 指标提取正则表达式
PATTERNS = {
    'micro_f1': r'micro_f1:\s*([\d.]+)±([\d.]+)',
    'macro_f1': r'macro_f1:\s*([\d.]+)±([\d.]+)',
    'greedy_results': r'\[CS-greedy\]\s*w=([\d.]+)\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)\s+Jaccard=([\d.]+)\s+size=([\d.]+)',
    'timing': r'\[timing\]\s*训练时间=([\d.]+)s.*测试时间=([\d.]+)s.*总运行时间=([\d.]+)s',
}


def extract_metrics_from_log(log_file):
    """从日志文件中提取指标"""
    if not log_file.exists():
        return None

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {}

    # 提取节点分类指标
    match = re.search(PATTERNS['micro_f1'], content)
    if match:
        metrics['micro_f1'] = f"{match.group(1)}±{match.group(2)}"
        metrics['micro_f1_val'] = float(match.group(1))

    match = re.search(PATTERNS['macro_f1'], content)
    if match:
        metrics['macro_f1'] = f"{match.group(1)}±{match.group(2)}"

    # 提取社区搜索结果（找最优 w）
    greedy_matches = re.findall(PATTERNS['greedy_results'], content)
    if greedy_matches:
        results = []
        for m in greedy_matches:
            w, p, r, f1, jaccard, size = [float(x) for x in m]
            results.append({
                'w': w, 'p': p, 'r': r, 'f1': f1,
                'jaccard': jaccard, 'size': size
            })
        best = max(results, key=lambda x: x['f1'])
        metrics['best_w'] = best['w']
        metrics['best_f1'] = best['f1']
        metrics['best_jaccard'] = best['jaccard']
        metrics['best_size'] = best['size']
        metrics['best_p'] = best['p']
        metrics['best_r'] = best['r']

    # 提取时间
    match = re.search(PATTERNS['timing'], content)
    if match:
        metrics['train_time'] = float(match.group(1))
        metrics['test_time'] = float(match.group(2))
        metrics['total_time'] = float(match.group(3))

    return metrics


def format_delta(value, baseline, unit=''):
    """格式化差值，带正负号"""
    delta = value - baseline
    sign = '+' if delta > 0 else ''
    return f"{value:.2f} ({sign}{delta:.2f}{unit})"


def print_comparison_table(results, dataset, baseline_key='full_model'):
    """打印对比表，按类别分组"""
    print(f"\n{'='*110}")
    print(f"消融实验结果对比 - {dataset}")
    print(f"{'='*110}\n")

    # 获取基线值
    baseline = results.get(baseline_key, {})
    baseline_f1 = baseline.get('best_f1', 0)
    baseline_jaccard = baseline.get('best_jaccard', 0)

    # 按类别分组
    by_category = defaultdict(list)
    for ablation_name, metrics in results.items():
        if ablation_name == baseline_key:
            continue
        cat = ABLATION_INFO.get(ablation_name, {}).get('category', '其他')
        by_category[cat].append((ablation_name, metrics))

    # 先打印基线
    print(f"{'类别':<12} {'消融实验':<22} {'节点分类':<16} {'社区 F1':<18} {'Jaccard':<18} {'w':<6} {'Size':<8}")
    print("-" * 110)

    if baseline_key in results:
        m = results[baseline_key]
        desc = ABLATION_INFO[baseline_key]['desc']
        print(f"{'基线':<12} {desc:<22} {m.get('micro_f1', '-'):<16} "
              f"{m.get('best_f1', 0):<18.2f} {m.get('best_jaccard', 0):<18.2f} "
              f"{m.get('best_w', 0):<6.1f} {m.get('best_size', 0):<8.0f}")

    print("-" * 110)

    # 按类别打印
    category_order = ['损失函数', '模块']
    for cat in category_order:
        if cat not in by_category:
            continue

        first_in_cat = True
        for ablation_name, metrics in sorted(by_category[cat],
                                              key=lambda x: x[1].get('best_f1', 0),
                                              reverse=True):
            info = ABLATION_INFO.get(ablation_name, {})
            desc = info.get('desc', ablation_name)

            node_cls = metrics.get('micro_f1', '-')

            # 计算差值
            f1 = metrics.get('best_f1', 0)
            jaccard = metrics.get('best_jaccard', 0)

            if baseline_f1 > 0:
                f1_str = format_delta(f1, baseline_f1)
                jaccard_str = format_delta(jaccard, baseline_jaccard)
            else:
                f1_str = f"{f1:.2f}"
                jaccard_str = f"{jaccard:.2f}"

            cat_str = cat if first_in_cat else ''
            first_in_cat = False

            print(f"{cat_str:<12} {desc:<22} {node_cls:<16} "
                  f"{f1_str:<28} {jaccard_str:<28} "
                  f"{metrics.get('best_w', 0):<6.1f} {metrics.get('best_size', 0):<8.0f}")

        print("-" * 110)

    print(f"{'='*110}\n")

    # 打印关键发现
    print("关键发现：\n")

    # 找出影响最大的消融
    impacts = []
    for ablation_name, metrics in results.items():
        if ablation_name == baseline_key:
            continue
        if 'best_f1' in metrics and baseline_f1 > 0:
            delta = metrics['best_f1'] - baseline_f1
            desc = ABLATION_INFO.get(ablation_name, {}).get('desc', ablation_name)
            impacts.append((desc, delta, ABLATION_INFO.get(ablation_name, {}).get('category', '')))

    if impacts:
        # 按影响排序（负值表示重要）
        impacts.sort(key=lambda x: x[1])

        print("按社区搜索 F1 影响排序（负值表示该组件重要）：\n")
        for desc, delta, cat in impacts:
            sign = '+' if delta > 0 else ''
            importance = "⚠️ 重要" if delta < -1.0 else "次要" if delta > 1.0 else "一般"
            print(f"  [{cat}] {desc:<25} {sign}{delta:.2f}  ({importance})")

        print()


def generate_summary_report(results, dataset, output_file):
    """生成 Markdown 格式的总结报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# 消融实验结果总结 - {dataset}\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 实验配置\n\n")
        f.write(f"- 数据集: {dataset}\n")
        f.write(f"- 训练轮数: 200\n")
        f.write(f"- 评估指标: Micro-F1 (节点分类), F1 & Jaccard (社区搜索)\n\n")

        # 获取基线值
        baseline = results.get('full_model', {})
        baseline_f1 = baseline.get('best_f1', 0)

        f.write("## 结果对比\n\n")

        # 按类别分组
        by_category = defaultdict(list)
        for ablation_name, metrics in results.items():
            cat = ABLATION_INFO.get(ablation_name, {}).get('category', '其他')
            by_category[cat].append((ablation_name, metrics))

        # 基线
        f.write("### 基线\n\n")
        f.write("| 实验 | 节点分类 Micro-F1 | 社区搜索 F1 | Jaccard | 最优 w | Size |\n")
        f.write("|------|-------------------|-------------|---------|--------|------|\n")

        if 'full_model' in results:
            m = results['full_model']
            desc = ABLATION_INFO['full_model']['desc']
            f.write(f"| {desc} | {m.get('micro_f1', '-')} | {m.get('best_f1', 0):.2f} | "
                   f"{m.get('best_jaccard', 0):.2f} | {m.get('best_w', 0):.1f} | "
                   f"{m.get('best_size', 0):.0f} |\n")

        f.write("\n")

        # 各类别
        category_order = ['损失函数', '模块']
        for cat in category_order:
            if cat not in by_category:
                continue

            f.write(f"### {cat}消融\n\n")
            f.write("| 实验 | 节点分类 Micro-F1 | 社区搜索 F1 (Δ) | Jaccard (Δ) | 最优 w | Size |\n")
            f.write("|------|-------------------|-------------------|-------------|--------|------|\n")

            for ablation_name, metrics in sorted(by_category[cat],
                                                  key=lambda x: x[1].get('best_f1', 0),
                                                  reverse=True):
                info = ABLATION_INFO.get(ablation_name, {})
                desc = info.get('desc', ablation_name)

                node_cls = metrics.get('micro_f1', '-')
                f1 = metrics.get('best_f1', 0)
                jaccard = metrics.get('best_jaccard', 0)

                # 计算差值
                if baseline_f1 > 0 and ablation_name != 'full_model':
                    delta_f1 = f1 - baseline_f1
                    delta_jaccard = jaccard - baseline.get('best_jaccard', 0)
                    sign_f1 = '+' if delta_f1 > 0 else ''
                    sign_j = '+' if delta_jaccard > 0 else ''
                    f1_str = f"{f1:.2f} ({sign_f1}{delta_f1:.2f})"
                    jaccard_str = f"{jaccard:.2f} ({sign_j}{delta_jaccard:.2f})"
                else:
                    f1_str = f"{f1:.2f}"
                    jaccard_str = f"{jaccard:.2f}"

                f.write(f"| {desc} | {node_cls} | {f1_str} | {jaccard_str} | "
                       f"{metrics.get('best_w', 0):.1f} | {metrics.get('best_size', 0):.0f} |\n")

            f.write("\n")

        f.write("## 分析\n\n")
        f.write("### 各组件贡献度\n\n")

        if baseline_f1 > 0:
            impacts = []
            for ablation_name, metrics in results.items():
                if ablation_name == 'full_model':
                    continue
                if 'best_f1' in metrics:
                    delta = metrics['best_f1'] - baseline_f1
                    desc = ABLATION_INFO.get(ablation_name, {}).get('desc', ablation_name)
                    cat = ABLATION_INFO.get(ablation_name, {}).get('category', '')
                    impacts.append((cat, desc, delta))

            impacts.sort(key=lambda x: x[2])

            f.write("按社区搜索 F1 影响排序（负值表示该组件重要）：\n\n")
            for cat, desc, delta in impacts:
                sign = '+' if delta > 0 else ''
                importance = "**重要**" if delta < -1.0 else "次要" if delta > 1.0 else "一般"
                f.write(f"- [{cat}] **{desc}**: {sign}{delta:.2f} ({importance})\n")

        f.write("\n### 结论\n\n")
        f.write("（根据实验结果填写各模块的重要性分析）\n")


def main():
    parser = argparse.ArgumentParser(description='收集消融实验结果')
    parser.add_argument('--results_dir', type=str, default='ablation/results',
                        help='结果目录')
    parser.add_argument('--dataset', type=str, default='ACM',
                        choices=['ACM', 'IMDB_NEW', 'DBLP'],
                        help='数据集名称')
    parser.add_argument('--output', type=str, default=None,
                        help='输出报告文件路径（Markdown 格式）')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[错误] 结果目录不存在: {results_dir}")
        sys.exit(1)

    # 收集所有结果
    results = {}
    for ablation_name in ABLATION_INFO.keys():
        log_file = results_dir / f"{ablation_name}_{args.dataset}.log"
        metrics = extract_metrics_from_log(log_file)
        if metrics:
            results[ablation_name] = metrics

    if not results:
        print(f"[警告] 未找到任何实验结果")
        print(f"请先运行: python ablation/run_ablations.py --dataset {args.dataset}")
        sys.exit(1)

    # 打印对比表
    print_comparison_table(results, args.dataset)

    # 生成报告
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / f"ablation_summary_{args.dataset}.md"

    generate_summary_report(results, args.dataset, output_path)
    print(f"\n[完成] 总结报告已保存到: {output_path}")


if __name__ == '__main__':
    main()
