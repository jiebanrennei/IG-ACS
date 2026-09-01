# 消融实验

本目录包含 HII-GNN 模型的消融实验代码，涵盖**损失函数**和**模块级**两个维度。

## 消融实验列表

### 损失函数消融
| 实验名称 | 描述 | 命令行参数 |
|---------|------|-----------|
| `no_intent_loss` | 去除意图一致性损失 | `--no_intent_loss` |
| `no_edge_feature_loss` | 去除边特征一致性损失 | `--no_edge_feature_loss` |

### 模块级消融
| 实验名称 | 描述 | 命令行参数 |
|---------|------|-----------|
| `no_rec_view` | 去除重构视图（候选边） | `--disable_candidate_edges` |
| `no_edge_importance` | 去除边重要性和可疑节点识别 | `--no_suspicious_kl --no_suspicious_boost` |

## 快速开始

### 列出所有消融实验
```bash
python ablation/run_ablations.py --list
```

### 运行所有消融实验
```bash
# ACM 数据集
python ablation/run_ablations.py --dataset ACM

# IMDB 数据集
python ablation/run_ablations.py --dataset IMDB_NEW
```

### 按类别运行
```bash
# 只运行损失函数消融
python ablation/run_ablations.py --dataset ACM --category 损失函数

# 只运行模块消融
python ablation/run_ablations.py --dataset ACM --category 模块
```

### 运行单个消融
```bash
# 去除意图损失
python ablation/run_ablations.py --dataset ACM --ablation no_intent_loss

# 去除重构视图
python ablation/run_ablations.py --dataset ACM --ablation no_rec_view
```

### 快速测试（100 轮）
```bash
python ablation/run_ablations.py --dataset ACM --epochs 100
```

### 收集和分析结果
```bash
# 收集 ACM 数据集的消融结果
python ablation/collect_results.py --dataset ACM

# 生成 Markdown 报告
python ablation/collect_results.py --dataset ACM --output ablation/results/summary.md
```

## 文件说明

- `run_ablations.py`: 主脚本，运行消融实验
  - `--list`: 列出所有可用实验
  - `--category`: 按类别运行（损失函数/模块）
  - `--ablation`: 运行单个实验
  - `--epochs`: 设置训练轮数
- `collect_results.py`: 收集实验结果并生成对比报告
  - 按类别分组显示结果
  - 自动计算与基线的差异
  - 生成 Markdown 总结报告
- `results/`: 实验结果目录（自动创建）
  - 每个实验生成一个 `.log` 文件
  - 自动生成 `ablation_summary_{dataset}.md` 总结报告

## 参数配置

不同数据集使用不同的默认配置：

| 数据集 | 编码器 | 隐藏层维度 | 特殊参数 |
|--------|--------|-----------|---------|
| ACM | hii | 128 | `--sparsify_topk 50 --hii_heads 4 --icra_heads 4 --icra_dim 128` |
| IMDB_NEW | gcn | 256 | 无 |
| DBLP | gcn | 256 | 无 |

## 注意事项

1. **显存需求**: ACM 数据集约需 10GB 显存，IMDB_NEW 约需 8GB
2. **运行时间**: 单个消融实验约需 30 分钟（200 轮训练）
3. **结果复现**: 所有实验使用固定随机种子（seed=123）
4. **日志文件**: 所有实验输出保存在 `results/` 目录下的 `.log` 文件中

## 已有的实验（不用重复）

以下实验已经在 `comprehensive_experiment_log.md` 中记录：
- ✅ 完整模型基线（GCN 和 HII 编码器）
- ✅ 不同 w 值的社区搜索扫描
- ✅ GCN vs HII 编码器对比
- ✅ Actor-Critic vs Greedy 对比
