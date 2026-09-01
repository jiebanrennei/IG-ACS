# 实验运行记录

记录每次实验的配置命令和结果，用于对比和复现。

---

## 2026-08-31 实验记录

### ACM + GCN (当前版本)

**命令**:
```bash
python train_ig.py --dataset ACM --encoder gcn --num_hidden 256 --num_epochs 200 > log/acm_gcn_memopt.log 2>&1
```

**结果**:
- 节点分类: micro_f1=89.78±0.48, macro_f1=89.78±0.50, acc=89.78±0.48
- CS-greedy 最优: w=0.3, P=63.17, R=76.92, F1=67.99, Jaccard=54.48, size=1631.9
- 训练时间: 175.83s

---

### ACM + GCN (base_model=GATConv)

**命令**:
```bash
python train_ig.py --dataset ACM --encoder gcn --base_model GATConv --num_hidden 256 --num_epochs 200
```

**结果**:
- 节点分类: micro_f1=89.25±0.91, macro_f1=89.13±0.97, acc=89.25±0.91
- CS-greedy 最优: w=0.3, P=60.27, R=78.92, F1=67.06, Jaccard=53.31, size=1726.8
- 训练时间: 248.05s (训练) + 520.27s (测试) = 769.06s (总)

**备注**: 使用 GATConv 作为基础模型，节点分类和社区搜索效果均略低于普通 GCN（F1: 67.06 vs 67.99）。

---

### ACM + HII (当前版本, sparsify_topk=50)

**命令**:
```bash
python train_ig.py --dataset ACM --encoder hii --sparsify_topk 50 --num_hidden 128 --num_proj_hidden 64 --num_edge_hidden 16 --num_layers 2 --hii_heads 4 --icra_heads 4 --icra_dim 128 --num_epochs 200 > log/acm_hii_memopt.log 2>&1
```

**结果**:
- 节点分类: micro_f1=90.94±0.47, macro_f1=91.03±0.43, acc=90.94±0.47
- CS-greedy 最优: w=0.2, P=65.95, R=76.61, F1=69.20, Jaccard=56.11, size=1507.0
- 训练时间: 300.26s

**备注**: 
- 必须加 `--sparsify_topk 50`，否则 PSP 关系（430万边）会导致 OOM
- PSP 稀疏化后: 4338213 -> 344296 edges (avg_deg 1079 -> 86)

---

### ACM + HII (topk=100)

**命令**:
```bash
python train_ig.py --dataset ACM --encoder hii --sparsify_topk 100 --num_hidden 128 --num_proj_hidden 64 --num_edge_hidden 16 --num_layers 2 --hii_heads 4 --icra_heads 4 --icra_dim 128 --num_epochs 200 > log/acm_hii_topk100.log 2>&1
```

**结果**:
- 节点分类: micro_f1=91.00±0.34, macro_f1=91.00±0.38, acc=91.00±0.34
- CS-greedy 最优: w=0.2, P=65.20, R=76.39, F1=69.04, Jaccard=56.22, size=1547.1
- 训练时间: 367.68s

**备注**: 与 topk=50 效果相近（F1: 69.04 vs 69.20），但训练更慢（368s vs 300s）。建议用 topk=50。

---

### DBLP + GCN (当前版本)

**命令**:
```bash
python train_ig.py --dataset DBLP --encoder gcn --num_hidden 256 --num_epochs 200 > log/dblp_gcn_memopt.log 2>&1
```

**结果**:
- 节点分类: micro_f1=80.20±1.13, macro_f1=79.38±1.21, acc=80.20±1.13
- CS-greedy 最优: w=0.3, P=45.06, R=60.89, F1=51.11, Jaccard=35.03, size=1379.2
- 训练时间: 73.80s

---

### DBLP + GCN (base_model=GATConv) 🚀

**命令**:
```bash
python train_ig.py --dataset DBLP --encoder gcn --base_model GATConv --num_hidden 256 --num_epochs 200
```

**结果**:
- 节点分类: micro_f1=79.10±0.53, macro_f1=78.36±0.49, acc=79.10±0.53
- CS-greedy 最优: w=0.3, P=46.12, R=66.03, F1=53.82, Jaccard=37.73, size=1445.2
- 训练时间: 76.15s (训练) + 529.76s (测试) = 607.42s (总)

**备注**: GATConv 在 DBLP 上效果优于普通 GCN！CS-F1 提升 2.71 个点（53.82 vs 51.11）。

---

### DBLP + HII (sparsify_topk=50) 🔴

**命令**:
```bash
python train_ig.py --dataset DBLP --encoder hii --sparsify_topk 50 --num_hidden 128 --num_proj_hidden 64 --num_edge_hidden 16 --num_layers 2 --hii_heads 4 --icra_heads 4 --icra_dim 128 --num_epochs 200 > log/dblp_hii_memopt.log 2>&1
```

**结果**:
- 节点分类: micro_f1=72.62±1.03, macro_f1=71.73±1.07, acc=72.62±1.03
- CS-greedy 最优: w=0.0, P=33.37, R=63.20, F1=43.37, Jaccard=27.95, size=1902.7
- 训练时间: 354.51s

**备注**: HII 在 DBLP 上效果大幅下降！比 GCN 低 7.74 个点（F1: 43.37 vs 51.11）。DBLP 建议继续用 GCN。

---

### IMDB_NEW + GCN (当前版本)

**命令**:
```bash
python train_ig.py --dataset IMDB_NEW --encoder gcn --num_hidden 256 --num_epochs 200 > log/imdb_gcn_memopt.log 2>&1
```

**结果**:
- 节点分类: micro_f1=待补充, macro_f1=待补充
- CS-greedy 最优: w=0.0, P=39.01, R=67.04, F1=49.20, Jaccard=33.20, size=2591.4
- 训练时间: 待补充

---

### IMDB_NEW + HII (当前版本) 🚀

**命令**:
```bash
python train_ig.py --dataset IMDB_NEW --encoder hii --num_hidden 256 --num_proj_hidden 128 --num_edge_hidden 16 --num_layers 2 --hii_heads 4 --icra_heads 4 --icra_dim 128 --num_epochs 200 > log/imdb_hii_memopt.log 2>&1
```

**结果**:
- 节点分类: micro_f1=89.70±0.62, macro_f1=89.69±0.62, acc=89.70±0.62
- CS-greedy 最优: w=0.0, P=50.37, R=71.67, F1=59.07, Jaccard=43.06, size=2140.8
- 训练时间: 374.22s

**备注**: HII 在 IMDB 上效果大幅提升！比 GCN 高 22 个点（节点分类）和 9.5 个点（社区搜索）。

---

### IMDB_NEW + GCN (base_model=GATConv)

**命令**:
```bash
python train_ig.py --dataset IMDB_NEW --encoder gcn --base_model GATConv --num_hidden 256 --num_epochs 200 > log/imdb_gat_memopt12.log 2>&1
```

**结果**:
- 节点分类: micro_f1=64.54±0.91, macro_f1=64.35±1.02, acc=64.54±0.91
- CS-greedy 最优: w=0.0, P=38.67, R=70.43, F1=49.79, Jaccard=33.61, size=2745.0
- 训练时间: 498.25s (训练) + 492.16s (测试) = 990.87s (总)

**备注**: GATConv 在 IMDB 上效果与 GCN 相近（F1: 49.79 vs 49.20），远不如 HII（59.07）。

---

## 历史最优结果 (best_results_summary.md, 2026-08-25)

### ACM + GCN (历史最优)

**结果**:
- 节点分类: micro_f1=90.53±0.51, macro_f1=90.56±0.55
- CS-greedy 最优: w=0.5, P=68.47, R=73.50, F1=69.42, Jaccard=56.98, size=1432.4
- 训练时间: 187.24s

### IMDB_NEW + GCN (历史最优)

**结果**:
- 节点分类: micro_f1=67.51±0.48, macro_f1=67.45±0.49
- CS-greedy 最优: w=0.0, P=39.32, R=67.36, F1=49.51, Jaccard=33.49, size=2583.4
- 训练时间: 440.60s

### DBLP + GCN (历史最优)

**结果**:
- 节点分类: micro_f1=80.18±0.64, macro_f1=79.45±0.74
- CS-greedy 最优: w=0.3, P=44.57, R=60.42, F1=50.65, Jaccard=34.53, size=1377.9
- 训练时间: 待补充

---

## 实验总结

### GCN vs HII 对比

| 数据集 | GCN F1 | HII F1 | HII 提升 | 结论 |
|--------|--------|--------|---------|------|
| **ACM** | 67.99 | 69.20 | +1.21 | ✅ HII 略有提升 |
| **IMDB** | 49.20 | **59.07** | **+9.87** | 🚀 HII 大幅提升 |
| **DBLP** | 51.11 | 43.37 | -7.74 | 🔴 HII 大幅下降 |

**结论**: HII 不是万能的，效果取决于数据集特性。
- IMDB: HII 非常适合（+9.87）
- ACM: HII 略有提升（+1.21）
- DBLP: HII 不适合，建议用 GCN

---

## 各数据集最优结果汇总

### ACM 数据集

| 方法 | 节点分类 F1 | CS-greedy 最优 w | CS P | CS R | CS F1 | CS Jaccard | size | 训练时间 |
|------|------------|-----------------|------|------|-------|-----------|------|---------|
| **GCN (历史最优)** | 90.53±0.51 | w=0.5 | 68.47 | 73.50 | **69.42** | 56.98 | 1432.4 | 187.24s |
| GCN (当前版本) | 89.78±0.48 | w=0.3 | 63.17 | 76.92 | 67.99 | 54.48 | 1631.9 | 175.83s |
| HII (topk=50) | **90.94±0.47** | w=0.2 | 65.95 | 76.61 | 69.20 | 56.11 | 1507.0 | 300.26s |
| GCN (GATConv) | 89.25±0.91 | w=0.3 | 60.27 | 78.92 | 67.06 | 53.31 | 1726.8 | 769.06s |

**ACM 最优配置**: GCN (历史最优), CS-F1=**69.42**

---

### DBLP 数据集

| 方法 | 节点分类 F1 | CS-greedy 最优 w | CS P | CS R | CS F1 | CS Jaccard | size | 训练时间 |
|------|------------|-----------------|------|------|-------|-----------|------|---------|
| **GCN (当前版本)** | **80.20±1.13** | w=0.3 | 45.06 | 60.89 | **51.11** | 35.03 | 1379.2 | 73.80s |
| GCN (历史最优) | 80.18±0.64 | w=0.3 | 44.57 | 60.42 | 50.65 | 34.53 | 1377.9 | - |
| HII (topk=50) | 72.62±1.03 | w=0.0 | 33.37 | 63.20 | 43.37 | 27.95 | 1902.7 | 354.51s |

**DBLP 最优配置**: GCN (当前版本), CS-F1=**51.11**

---

### IMDB_NEW 数据集

| 方法 | 节点分类 F1 | CS-greedy 最优 w | CS P | CS R | CS F1 | CS Jaccard | size | 训练时间 |
|------|------------|-----------------|------|------|-------|-----------|------|---------|
| **HII** | **89.70±0.62** | w=0.0 | 50.37 | 71.67 | **59.07** | 43.06 | 2140.8 | 374.22s |
| GCN (历史最优) | 67.51±0.48 | w=0.0 | 39.32 | 67.36 | 49.51 | 33.49 | 2583.4 | 440.60s |
| GCN (当前版本) | - | w=0.0 | 39.01 | 67.04 | 49.20 | 33.20 | 2591.4 | - |

**IMDB_NEW 最优配置**: HII, CS-F1=**59.07**

---

## 最终推荐配置

| 数据集 | 推荐方法 | 节点分类 F1 | CS-F1 | 推荐命令 |
|--------|---------|------------|-------|---------|
| **ACM** | GCN (历史最优) | 90.53 | **69.42** | `python train_ig.py --dataset ACM --encoder gcn --num_hidden 256 --num_epochs 200` |
| **DBLP** | **GCN (GATConv)** | 79.10 | **53.82** | `python train_ig.py --dataset DBLP --encoder gcn --base_model GATConv --num_hidden 256 --num_epochs 200` |
| **IMDB_NEW** | HII | 89.70 | **59.07** | `python train_ig.py --dataset IMDB_NEW --encoder hii --num_hidden 256 --num_epochs 200` |

---

## 方法对比总结

### GCN vs HII vs GATConv

| 数据集 | GCN F1 | HII F1 | GATConv F1 | 最优方法 | 提升幅度 |
|--------|--------|--------|-----------|---------|---------|
| **ACM** | 69.42 | 69.20 | 67.06 | GCN | - |
| **IMDB** | 49.51 | **59.07** | 49.79 | **HII** | **+9.56** |
| **DBLP** | 51.11 | 43.37 | **53.82** | **GATConv** | **+2.71** |

**关键发现**:
1. **IMDB**: HII 大幅优于 GCN（+9.56），是最佳选择
2. **DBLP**: GCN 大幅优于 HII（+7.74），HII 不适合
3. **ACM**: GCN 和 HII 效果接近，GCN 略优且更快
4. **GATConv**: 在 ACM 上效果不如普通 GCN，不推荐使用

---

## 待运行实验

- [ ] Twitter + GCN (需先下载数据)
- [ ] Twitter + HII
- [ ] 其他数据集的超参数调优
