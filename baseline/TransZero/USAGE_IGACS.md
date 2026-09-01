# 使用 TransZero 基线方法

## 概述

TransZero 是 VLDB 2024 论文 "Efficient Unsupervised Community Search with Pre-trained Graph Transformer" 的代码实现。

**论文**: Efficient Unsupervised Community Search with Pre-trained Graph Transformer  
**发表**: VLDB 2024

## 快速开始

### 1. 转换数据集

首先将 IG-ACS 的数据集转换为 TransZero 格式：

```bash
cd baseline/TransZero

# 转换所有数据集 (ACM, DBLP, IMDB_NEW)
python convert_igacs_to_transzero.py --dataset all --output_dir dataset

# 或只转换单个数据集
python convert_igacs_to_transzero.py --dataset ACM --output_dir dataset
python convert_igacs_to_transzero.py --dataset DBLP --output_dir dataset
python convert_igacs_to_transzero.py --dataset IMDB_NEW --output_dir dataset
```

转换后会生成：
- `dataset/acm/acm.pt` - 图数据（邻接矩阵 + 特征 + 标签）
- `dataset/acm/acm.edges` - 边列表
- `dataset/acm/acm.query` - 查询节点（150个）
- `dataset/acm/acm.gt` - 真实社区

### 2. 预训练模型

```bash
# ACM
python link_pretrain.py --dataset acm --batch_size 4019 --dropout 0.1 --hidden_dim 512 --hops 5 --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01 --weight_decay=1e-05 --epochs 100

# DBLP
python link_pretrain.py --dataset dblp --batch_size 4057 --dropout 0.1 --hidden_dim 512 --hops 5 --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01 --weight_decay=1e-05 --epochs 100

# IMDB_NEW
python link_pretrain.py --dataset imdb_new --batch_size 4019 --dropout 0.1 --hidden_dim 512 --hops 5 --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01 --weight_decay=1e-05 --epochs 100
```

**注意**: `--batch_size` 需要设置为对应数据集的节点数

### 3. 测试（社区搜索）

```bash
# 全局搜索（二分查找）
python accuracy_globalsearch.py

# 局部搜索
python accuracy_localsearch.py
```

### 4. 批量运行

```bash
# 训练所有数据集
bash ./training_all.sh

# 测试所有数据集
bash ./test_all_global.sh >> ./logs/test_all_global.txt 2>&1 &
bash ./test_all_local.sh >> ./logs/test_all_local.txt 2>&1 &
```

## 数据集说明

| 数据集 | 节点数 | 边数 | 类别数 | 说明 |
|--------|--------|------|--------|------|
| ACM | 4019 | ~4.3M | 3 | 异构图合并（PAP+PSP） |
| DBLP | 4057 | ~5M | 4 | 异构图合并（APA+APCPA） |
| IMDB_NEW | 4780 | ~162K | 3 | 异构图合并（MAM+MDM） |

**注意**: 由于原始数据集是异构图，转换时会合并所有 meta-path 为同构图。这可能会导致边数增加。

## 参数调优

如果效果不好，可以尝试调整：

- `--hops`: 邻居采样跳数（默认 5）
- `--hidden_dim`: 隐藏层维度（默认 512）
- `--n_heads`: 注意力头数（默认 8）
- `--n_layers`: Transformer 层数（默认 1）
- `--epochs`: 训练轮数（默认 100）

## 预期结果

TransZero 在同构图上效果较好。对于你的异构图数据：

**优点**:
- Graph Transformer 能捕捉长距离依赖
- 无监督预训练，不需要标签

**缺点**:
- 合并 meta-path 可能丢失异构信息
- 计算复杂度高（Transformer 是 O(n²)）

## 对比实验建议

1. 在 ACM/DBLP/IMDB 上运行 TransZero
2. 记录 F1、Jaccard、Precision、Recall
3. 与你的方法（HII-GNN）对比
4. 分析 TransZero 在异构图上的表现

## 常见问题

**Q: 为什么边数这么多？**
A: 因为合并了所有 meta-path。例如 ACM 的 PSP 有 430 万条边。

**Q: 可以只用部分 meta-path 吗？**
A: 可以修改 `convert_igacs_to_transzero.py` 中的 `meta_path='all'` 参数。

**Q: OOM 怎么办？**
A: 减小 `--batch_size` 或 `--hidden_dim`，或使用 `--hops 3`。

## 参考

```bibtex
@article{wang2024efficient,
  title={Efficient Unsupervised Community Search with Pre-trained Graph Transformer},
  author={Wang, Jianwei and Wang, Kai and Lin, Xuemin and Zhang, Wenjie and Zhang, Ying},
  journal={arXiv preprint arXiv:2403.18869},
  year={2024}
}
```
