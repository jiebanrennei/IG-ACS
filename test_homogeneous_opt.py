#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
同构图性能优化测试脚本

核心优化：
1. 调整对比学习温度参数（tau）
2. 增加候选边数量
3. 使用多种候选边来源
4. 调整损失权重

使用方法：
    python test_homogeneous_opt.py --dataset com-amazon --encoder gcn --num_hidden 256 --num_epochs 200
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from datetime import datetime

# 导入必要的模块
from utils import set_everything, get_dataset
from train_ig import train_model


def get_homogeneous_config(dataset_name):
    """
    为同构图数据集返回优化的配置参数
    """
    # 基础配置（适用于所有同构图）
    config = {
        # 对比学习温度：更低的温度增强区分度
        'tau': 0.25,  # 默认 0.4，同构图用 0.25

        # 候选边配置：增加重构能力
        'num_cand_per_node': 10,  # 默认 5，增加到 10
        'cand_sources': 'embed,twohop,common',  # 使用多种来源
        'cand_label_mode': 'soft',

        # 损失权重：增强对比学习，适当降低重构
        'lambda_cand_bce': 0.1,  # 候选边重构损失权重（从 0 增加到 0.1）
        'lambda_rec': 0.05,  # 对抗特征保持损失（从 0.1 降低到 0.05）
        'adv_lambda': 1.2,  # 对抗损失权重（从 1.0 增加到 1.2）

        # 边权重生成
        'adv_temp': 0.8,  # 从 1.0 降低到 0.8，使边权重更尖锐
        'bias': 0.001,  # 从 0.0001 增加到 0.001

        # 训练参数
        'learning_rate_train': 0.001,
        'learning_rate_adv': 0.0005,
        'wd_train': 1e-5,
        'wd_adv': 1e-5,
    }

    # 针对特定数据集的微调
    if dataset_name == 'com-amazon':
        config.update({
            'num_hidden': 256,
            'num_proj_hidden': 128,
            'num_edge_hidden': 32,
            'num_layers': 2,
            'num_cand_per_node': 15,  # Amazon 社区较大，增加候选边
        })
    elif dataset_name == 'com-dblp':
        config.update({
            'num_hidden': 256,
            'num_proj_hidden': 128,
            'num_edge_hidden': 32,
            'num_layers': 2,
        })
    elif dataset_name == 'com-youtube':
        config.update({
            'num_hidden': 128,  # 降低隐藏层维度避免 OOM
            'num_proj_hidden': 64,
            'num_edge_hidden': 16,
            'num_layers': 2,
            'minibatch': True,  # 启用 mini-batch 训练
            'minibatch_size': 2048,
        })
    elif dataset_name == 'com-twitter':
        config.update({
            'num_hidden': 128,
            'num_proj_hidden': 64,
            'num_edge_hidden': 16,
            'num_layers': 2,
            'minibatch': True,
            'minibatch_size': 2048,
        })
    elif dataset_name == 'com-livejournal':
        config.update({
            'num_hidden': 128,
            'num_proj_hidden': 64,
            'num_edge_hidden': 16,
            'num_layers': 2,
            'minibatch': True,
            'minibatch_size': 2048,
        })

    return config


def main():
    parser = argparse.ArgumentParser(description='同构图性能优化测试')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['com-amazon', 'com-dblp', 'com-youtube', 'com-twitter', 'com-livejournal'],
                        help='同构图数据集名称')
    parser.add_argument('--encoder', type=str, default='gcn',
                        choices=['gcn', 'hii'],
                        help='编码器类型（同构图推荐 gcn）')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--seed', type=int, default=123,
                        help='随机种子')
    parser.add_argument('--use_optimization', action='store_true', default=True,
                        help='使用优化配置（默认开启）')
    parser.add_argument('--query', type=str,
                        default='找出在社交网络上通过隐蔽连接协同的群体',
                        help='查询文本')

    args = parser.parse_args()

    # 设置随机种子
    set_everything(args.seed)

    # 获取优化的配置
    if args.use_optimization:
        opt_config = get_homogeneous_config(args.dataset)
        print(f"\n{'='*60}")
        print(f"同构图优化配置 ({args.dataset})")
        print(f"{'='*60}")
        for k, v in opt_config.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}\n")
    else:
        opt_config = {}
        print("\n使用默认配置（未优化）\n")

    # 构建训练参数
    train_args = argparse.Namespace(
        dataset=args.dataset,
        encoder=args.encoder,
        num_epochs=args.num_epochs,
        seed=args.seed,
        query=args.query,
        intent_source='random',  # 同构图使用随机意图
        # 合并优化配置
        **opt_config
    )

    # 添加必要的默认参数
    default_params = {
        'perturbed_data': None,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'intent_dim': 256,
        'lambda_intent': 0.3,
        'intent_num_queries': 100,
        'minibatch_num_neighbors': 10,
        'minibatch_num_batches': 10,
        'intent_encoder_name': 'paraphrase-multilingual-MiniLM-L12-v2',
        'intent_library_path': None,
        'meta_path': None,
        'cand_source_topk': None,
        'cand_intent_dist_k': 16,
        'cand_intent_dist_tau': 0.2,
        'cand_hard_threshold': 0.5,
        'disable_candidate_edges': False,
        'cand_refresh_interval': 20,
        'n2v_walk_length': 20,
        'n2v_context_size': 10,
        'n2v_walks_per_node': 10,
        'n2v_p': 1.0,
        'n2v_q': 1.0,
        'n2v_epochs': 5,
        'hii_heads': 4,
        'top_k_suspicious': 50,
        'suspicious_boost': 1.5,
        'cs_num_queries': 40,
        'query_file': None,
        'use_actor_critic': False,
        'ac_epochs': 100,
        'ac_lr': 1e-3,
        'ac_max_size': 200,
        'ac_size_sweep': None,
        'resume': False,
        'eval_only': False,
        'model_name': None,
        'ckpt_path': None,
        'ckpt_interval': 1,
        'icra_heads': 4,
        'icra_dim': 128,
        'relation_fusion': 'icra',
        'cs_relations': None,
        'lambda_rel_entropy': 0.0,
        'sparsify_topk': None,
        'cs_full_graph': True,
        'filename_prefix': './datasets',
        'log_dir': './log',
    }

    # 合并默认参数
    for k, v in default_params.items():
        if not hasattr(train_args, k):
            setattr(train_args, k, v)

    # 运行训练
    print(f"\n开始训练 {args.dataset}...")
    print(f"编码器: {args.encoder}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"随机种子: {args.seed}\n")

    try:
        train_model(train_args)
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
