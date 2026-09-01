#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验主脚本
包含损失函数和模块级消融实验
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# 消融实验配置
ABLATION_CONFIGS = {
    # ===== 损失函数消融 =====
    "no_intent_loss": {
        "description": "去除意图一致性损失",
        "category": "损失函数",
        "extra_args": ["--no_intent_loss"],
    },
    "no_edge_feature_loss": {
        "description": "去除边特征一致性损失",
        "category": "损失函数",
        "extra_args": ["--no_edge_feature_loss"],
    },

    # ===== 模块级消融 =====
    "no_rec_view": {
        "description": "去除重构视图（候选边）",
        "category": "模块",
        "extra_args": ["--disable_candidate_edges"],
    },
    "no_edge_importance": {
        "description": "去除边重要性和可疑节点识别",
        "category": "模块",
        "extra_args": ["--no_suspicious_kl", "--no_suspicious_boost"],
    },
}


def get_default_config(dataset):
    """根据数据集获取默认配置"""
    if dataset == "ACM":
        return {
            "encoder": "hii",
            "hidden": 128,
            "extra_base_args": [
                "--sparsify_topk", "50",
                "--hii_heads", "4",
                "--icra_heads", "4",
                "--icra_dim", "128",
            ],
        }
    elif dataset in ["IMDB_NEW", "DBLP"]:
        return {
            "encoder": "gcn",
            "hidden": 256,
            "extra_base_args": [],
        }
    else:
        raise ValueError(f"未知数据集: {dataset}")


def run_ablation(dataset, ablation_name, config, results_dir, epochs=200):
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"消融实验: {config['description']}")
    print(f"类别: {config.get('category', '未分类')}")
    print(f"数据集: {dataset}")
    print(f"{'='*60}\n")

    # 构建命令
    cmd = [
        sys.executable, "train_ig.py",
        "--dataset", dataset,
        "--encoder", config.get("encoder", "gcn"),
        "--num_hidden", str(config.get("hidden", 256)),
        "--num_epochs", str(epochs),
        "--model_name", f"ablation_{ablation_name}",
    ]

    # 添加数据集特定的基础参数
    cmd.extend(config.get("extra_base_args", []))

    # 添加消融特定的参数
    cmd.extend(config.get("extra_args", []))

    # 日志文件
    log_file = results_dir / f"{ablation_name}_{dataset}.log"

    print(f"执行命令: {' '.join(cmd)}")
    print(f"日志文件: {log_file}")

    # 执行命令
    with open(log_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # 实时输出并写入日志
        for line in process.stdout:
            print(line, end='')
            f.write(line)

        process.wait()

    if process.returncode != 0:
        print(f"\n[错误] 实验失败，返回码: {process.returncode}")
        return False

    print(f"\n[完成] 实验成功，结果保存在: {log_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description='运行消融实验')
    parser.add_argument('--dataset', type=str, default='ACM',
                        choices=['ACM', 'IMDB_NEW', 'DBLP'],
                        help='数据集名称')
    parser.add_argument('--ablation', type=str, default=None,
                        help=f'指定消融实验名称。不指定则运行所有实验')
    parser.add_argument('--category', type=str, default=None,
                        choices=['损失函数', '模块'],
                        help='只运行某个类别的消融实验')
    parser.add_argument('--results_dir', type=str, default='ablation/results',
                        help='结果保存目录')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数（默认 200）')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用的消融实验')

    args = parser.parse_args()

    # 列出所有消融实验
    if args.list:
        print("\n可用的消融实验：\n")
        by_category = {}
        for name, config in ABLATION_CONFIGS.items():
            cat = config.get('category', '未分类')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((name, config['description']))

        for cat, experiments in by_category.items():
            print(f"\n【{cat}】")
            for name, desc in experiments:
                print(f"  {name:<25} - {desc}")
        print()
        return

    # 创建结果目录
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# 消融实验开始")
    print(f"# 数据集: {args.dataset}")
    print(f"# 训练轮数: {args.epochs}")
    print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# 结果目录: {results_dir}")
    print(f"{'#'*60}\n")

    # 获取默认配置
    base_config = get_default_config(args.dataset)

    # 确定要运行的消融实验
    if args.ablation:
        # 运行单个消融实验
        if args.ablation not in ABLATION_CONFIGS:
            print(f"[错误] 未知的消融实验: {args.ablation}")
            print(f"使用 --list 查看所有可用实验")
            sys.exit(1)

        config = base_config.copy()
        config.update(ABLATION_CONFIGS[args.ablation])

        success = run_ablation(args.dataset, args.ablation, config, results_dir, args.epochs)
        sys.exit(0 if success else 1)
    elif args.category:
        # 运行某个类别的所有消融
        experiments = [(name, cfg) for name, cfg in ABLATION_CONFIGS.items()
                      if cfg.get('category') == args.category]

        if not experiments:
            print(f"[错误] 类别 '{args.category}' 下没有实验")
            sys.exit(1)

        print(f"\n运行类别: {args.category} ({len(experiments)} 个实验)\n")

        results = {}
        for ablation_name, ablation_config in experiments:
            config = base_config.copy()
            config.update(ablation_config)
            success = run_ablation(args.dataset, ablation_name, config, results_dir, args.epochs)
            results[ablation_name] = success

        # 打印总结
        print(f"\n{'#'*60}")
        print(f"# 类别 {args.category} 消融实验总结")
        print(f"{'#'*60}")
        for name, success in results.items():
            status = "✓ 成功" if success else "✗ 失败"
            print(f"{status}: {ABLATION_CONFIGS[name]['description']}")

        print(f"\n所有结果保存在: {results_dir}")
    else:
        # 运行所有消融实验
        results = {}
        for ablation_name, ablation_config in ABLATION_CONFIGS.items():
            config = base_config.copy()
            config.update(ablation_config)

            success = run_ablation(args.dataset, ablation_name, config, results_dir, args.epochs)
            results[ablation_name] = success

        # 打印总结
        print(f"\n{'#'*60}")
        print(f"# 消融实验总结")
        print(f"{'#'*60}")

        # 按类别分组显示
        by_category = {}
        for name, success in results.items():
            cat = ABLATION_CONFIGS[name].get('category', '未分类')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((name, success))

        for cat, experiments in by_category.items():
            print(f"\n【{cat}】")
            for name, success in experiments:
                status = "✓" if success else "✗"
                print(f"  {status} {ABLATION_CONFIGS[name]['description']}")

        print(f"\n所有结果保存在: {results_dir}")
        print(f"\n下一步：python ablation/collect_results.py --dataset {args.dataset}")


if __name__ == '__main__':
    main()
