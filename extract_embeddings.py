#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取训练好的模型节点嵌入，用于可视化分析
使用方法：
    python extract_embeddings.py --dataset ACM --checkpoint checkpoints/xxx.pt
"""

import argparse
import os
import torch
import numpy as np
from utils import get_dataset, set_everything
from model import Model


def extract_embeddings(args):
    """提取并保存节点嵌入"""

    # 设置随机种子
    set_everything(args.seed)

    # 加载数据集
    print(f"[INFO] 加载数据集: {args.dataset}")
    data, dataset = get_dataset(args.dataset, args.filename_prefix)

    # 检测图类型
    use_multi = (args.encoder == 'hii') and getattr(data, 'num_relations', 1) > 1

    # 加载模型
    print(f"[INFO] 加载模型: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # 创建模型实例
    model = Model(
        num_nodes=data.num_nodes,
        num_edges=data.num_edges if hasattr(data, 'num_edges') else data.edge_index.size(1),
        in_channels=dataset.num_features,
        num_hidden=args.num_hidden,
        num_classes=dataset.num_classes if hasattr(dataset, 'num_classes') else None,
        encoder=args.encoder,
        base_model=args.base_model,
        use_multi=use_multi,
        num_relations=getattr(data, 'num_relations', 1)
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 提取嵌入
    print(f"[INFO] 提取节点嵌入...")
    with torch.no_grad():
        embeddings = model.get_embeddings(data)

        # 如果是元组，取第一个
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]

        embeddings = embeddings.cpu().numpy()

    print(f"[INFO] 嵌入形状: {embeddings.shape}")

    # 保存嵌入
    output_path = f"embeddings_{args.dataset}.npy"
    np.save(output_path, embeddings)
    print(f"[INFO] 嵌入已保存到: {output_path}")

    # 如果有节点标签，也保存标签
    if hasattr(data, 'y') and data.y is not None:
        labels = data.y.cpu().numpy()
        label_path = f"labels_{args.dataset}.npy"
        np.save(label_path, labels)
        print(f"[INFO] 标签已保存到: {label_path}")

    print(f"[INFO] 完成！现在可以使用 vis.py 进行可视化：")
    print(f"  EMBEDDING_PATH = '{output_path}'")
    print(f"  LABEL_PATH = 'labels_{args.dataset}.npy'  # 可选")


def main():
    parser = argparse.ArgumentParser(description='提取节点嵌入用于可视化')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称 (ACM/DBLP/IMDB)')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--encoder', type=str, default='hii', help='编码器类型')
    parser.add_argument('--base_model', type=str, default='GCNConv', help='基础模型')
    parser.add_argument('--num_hidden', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--filename_prefix', type=str, default='./datasets', help='数据集路径前缀')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')

    args = parser.parse_args()
    extract_embeddings(args)


if __name__ == '__main__':
    main()
