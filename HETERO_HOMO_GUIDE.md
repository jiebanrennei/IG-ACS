# IG-ACS 异构/同构图实验指南

## 📊 支持的数据集

### 异构图数据集 (Heterogeneous Graphs)
- **ACM**: 学术网络，包含 paper-author, paper-subject 关系
- **DBLP**: 学术网络，包含多种关系类型
- **IMDB_NEW**: 电影网络，包含 movie-actor, movie-director 关系

### 同质图数据集 (Homogeneous Graphs)
从 SLRL (AAAI 2025) 转换而来，用于基线对比：
- **com-amazon**: 33K 节点, 34K 边, 4,517 社区
- **com-dblp**: 314K 节点, 467K 边, 4,559 社区
- **com-youtube**: 1.1M 节点, 1.4M 边, 2,865 社区
- **com-twitter**: 1.1M 节点, 1.3M 边, 2,838 社区
- **com-livejournal**: 3.8M 节点, 4.9M 边, 4,510 社区

## 🚀 快速开始

### 1. 快速验证（小参数测试）
```bash
chmod +x quick_validate.sh
./quick_validate.sh
```

### 2. 完整实验
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

### 3. 单独运行

#### 异构图
```bash
# ACM
python train_ig.py --dataset ACM --encoder hii --num_hidden 256 --num_epochs 200 --sparsify_topk 50

# DBLP
python train_ig.py --dataset DBLP --encoder gcn --base_model GATConv --num_hidden 256 --num_epochs 200

# IMDB
python train_ig.py --dataset IMDB_NEW --encoder hii --num_hidden 256 --num_epochs 200
```

#### 同质图
```bash
# 小数据集（推荐先测试）
python train_ig.py --dataset com-amazon --encoder gcn --num_hidden 256 --num_epochs 200
python train_ig.py --dataset com-dblp --encoder gcn --num_hidden 256 --num_epochs 200

# 中等数据集
python train_ig.py --dataset com-youtube --encoder gcn --num_hidden 256 --num_epochs 200
python train_ig.py --dataset com-twitter --encoder gcn --num_hidden 256 --num_epochs 200

# 大数据集（需要大内存）
python train_ig.py --dataset com-livejournal --encoder gcn --num_hidden 256 --num_epochs 200
```

## 🔬 与 SLRL 对比实验

### 实验设计
在你的论文中，你可以展示：

1. **同质图对比**（你的方法 vs SLRL）
   - com-amazon
   - com-dblp
   - com-youtube
   - com-twitter
   - com-livejournal

2. **异构图实验**（展示你的方法的优势）
   - ACM
   - DBLP
   - IMDB_NEW
   - **SLRL 无法运行**（因为不支持异构图）

### 运行 SLRL
```bash
cd /path/to/AAAI2025-SLRL-main
python mainSLRL.py
```

SLRL 会依次运行：amazon, dblp, twitter, youtube, lj

### 结果对比表格示例

**Table 1: Homogeneous Graph Results**

| Method | com-amazon (F1) | com-dblp (F1) | com-youtube (F1) |
|--------|----------------|---------------|------------------|
| SLRL | xx.xx | xx.xx | xx.xx |
| Ours | **xx.xx** | **xx.xx** | **xx.xx** |

**Table 2: Heterogeneous Graph Results**

| Method | ACM (F1) | DBLP (F1) | IMDB (F1) |
|--------|----------|-----------|-----------|
| GCN | xx.xx | xx.xx | xx.xx |
| HII | xx.xx | xx.xx | xx.xx |
| Ours | **xx.xx** | **xx.xx** | **xx.xx** |

*Note: SLRL cannot run on heterogeneous graphs*

## 📝 实验记录

所有实验结果记录在 `best_per/experiment_log.md` 中。

## 🔧 技术细节

### 自动检测图类型
代码会自动检测图类型：
```python
use_multi = (args.encoder == 'hii') and getattr(data, 'num_relations', 1) > 1
```

- 异构图（num_relations > 1）：使用多关系融合模块
- 同质图（num_relations = 1）：使用单关系处理

### 数据集转换
使用 `convert_slrl_datasets.py` 将 SLRL 数据集转换为 IG-ACS 格式：
```bash
python convert_slrl_datasets.py
```

转换后的数据集位于 `datasets/` 目录下：
- `com-Amazon/`
- `com-DBLP/`
- `com-Youtube/`
- `com-LiveJournal/`
- `com-Twitter/`

## 💡 建议

1. **先在小数据集上测试**：com-amazon（最小）
2. **逐步增加数据规模**：com-dblp -> com-youtube -> com-twitter -> com-livejournal
3. **记录所有结果**：更新 `best_per/experiment_log.md`
4. **对比 SLRL**：在相同同质图数据集上运行 SLRL，对比性能

## ⚠️ 注意事项

1. **内存需求**：
   - com-amazon: ~500MB ✅
   - com-dblp: ~2GB ✅
   - com-youtube: ~4GB ⚠️
   - com-twitter: ~4GB ⚠️
   - com-livejournal: ~8GB 🔴

2. **GPU 内存**：
   - 推荐使用 RTX 3090 (24GB) 或更大显存
   - 大数据集可能需要 CPU offload

3. **训练时间**：
   - 小数据集：几分钟
   - 大数据集：可能需要数小时
