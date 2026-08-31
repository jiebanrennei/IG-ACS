# IG-ACS：基于意图引导的对抗性社区搜索

本项目是在 EDA-GCL 图对比学习框架基础上的研究扩展，面向网络中的隐蔽协同群体、对抗社区和潜在异常关系发现。

给定观测图、查询节点和自然语言查询，模型学习与查询意图相关的节点表示，模拟观测边的删减，并重构可能被隐藏的连接，最终搜索出与查询目标相关的对抗性社区。

> 当前仓库同时保留了原始 EDA-GCL 基线、IG-ACS 扩展、数据扰动脚本、实验日志和论文材料。根目录的 `train_ig.py` 是 IG-ACS 的主要实验入口。

## 研究内容

IG-ACS 的整体流程如下：

```text
图数据、节点特征、查询节点、查询文本
                  │
                  ▼
          查询意图向量生成
                  │
                  ▼
       HII-GNN 意图感知节点编码
                  │
                  ▼
       意图引导的边权重学习
             │            │
             ▼            ▼
       对抗视图 G_adv   重构视图 G_rec
             │            │
             └──────┬─────┘
                    ▼
             双视图图对比学习
                    │
                    ▼
          可疑节点识别与社区搜索
                    │
                    ▼
       可选 Actor-Critic 社区构建器
```

### 核心模块

- **IG-ESAA**：将查询意图注入边权重模型，使边的删减和保留具有目标导向性。
- **AR-DVCL**：构造互补的对抗视图和重构视图。前者模拟删边或稀疏化，后者从候选缺失边中恢复潜在连接。
- **HII-GNN**：在局部边级、邻域聚合和全局融合三个层次注入意图信息。
- **边重要性学习**：综合拓扑、语义和意图相关性识别重要边及可疑节点。
- **社区搜索**：支持相似度搜索、贪心搜索、动态查询搜索和 Actor-Critic 社区扩展。

## 目录结构

```text
train_ig.py                    IG-ACS 主训练与评估入口
ig_model.py                   意图引导边模型、双视图模型和意图生成模块
hii_gnn.py                    层次化意图注入 GNN
edge_importance.py            可疑节点与边重要性计算
actor_critic.py               Actor-Critic 社区构建器
multi_relation_fusion.py      多关系/多元路径融合编码器
adversarial_intent_encoder.py 可选的文本意图编码器
eval.py                       节点分类、社区搜索和评估指标
utils.py                      数据集加载、预处理和通用工具

train_homo.py                 EDA-GCL 同质图基线
train_hete.py                 EDA-GCL 异配图基线
run_batch.py                  按 JSON 配置批量运行 train_ig.py
perturb_datasets.py           数据扰动与对抗实验脚本
data/                         数据集、扰动数据和数据处理脚本
script/                       原始基线训练脚本
log/                          实验日志
article/                      论文、调研报告和方法设计材料
```

## 数据集

`utils.py` 中已注册的社区搜索数据集包括：

- `ACM`
- `DBLP`
- `IMDB`
- `IMDB_NEW`
- `com-amazon`
- `com-dblp`
- `com-youtube`

此外还保留了 EDA-GCL 使用的节点分类数据集，例如 `cora_lcc`、`citeseer_lcc`、`chameleon`、`squirrel`、`Actor`、`Texas` 和 `Wisconsin`。

社区搜索数据通常包含节点特征、节点类型、标签、边关系或基于 meta-path 的邻接矩阵。没有节点属性的数据集会使用图结构特征作为输入。

## 安装环境

推荐使用 Python 3.10，并根据本机环境安装 PyTorch、PyTorch Geometric 及其依赖。项目主要依赖包括：

- PyTorch 2.0+
- PyTorch Geometric 2.6.1
- NumPy 1.26.4
- SciPy
- scikit-learn 1.6.1
- PyYAML 6.0.2
- DeepRobust 0.2.11
- tqdm

可以使用 Docker 构建 CPU 环境：

```bash
docker build -t ig-acs .
docker run --rm ig-acs
```

也可以直接安装依赖：

```bash
pip install -r requirements.txt
```

注意：当前 `requirements.txt` 中存在几个拼写错误，直接安装前请将以下包名修正：

```text
torc        -> torch
PyYAM       -> PyYAML
scikit-lear -> scikit-learn
```

不同 CUDA、PyTorch 和 PyTorch Geometric 版本可能导致安装命令不同，建议先按 PyTorch 版本选择对应的 PyG 依赖。

## 运行 IG-ACS

最小训练示例：

```bash
python train_ig.py \
    --dataset ACM \
    --encoder hii \
    --intent_source random \
    --num_hidden 256 \
    --num_epochs 200
```

使用查询文本编码意图时：

```bash
python train_ig.py \
    --dataset ACM \
    --encoder hii \
    --intent_source encoder \
    --query "找出与查询目标相关的隐蔽协同群体" \
    --num_hidden 256 \
    --num_epochs 200
```

`--intent_source encoder` 需要额外安装并配置文本编码器；如果编码器不可用，训练脚本会回退到固定随机意图向量。默认的 `random` 模式不依赖文本模型，适合先验证训练流程。

启用 Actor-Critic 社区搜索：

```bash
python train_ig.py \
    --dataset ACM \
    --encoder hii \
    --intent_source random \
    --use_actor_critic \
    --ac_epochs 100
```

使用扰动数据进行鲁棒性实验：

```bash
python train_ig.py \
    --dataset ACM \
    --perturbed_data <path-to-perturbed-data> \
    --encoder hii
```

实际可用参数较多，可通过以下命令查看：

```bash
python train_ig.py --help
```

## 批量实验

`run_batch.py` 根据 JSON 配置批量调用 `train_ig.py`，支持选择数据集、实验 profile、参数覆盖和独立日志目录：

```bash
python run_batch.py --config batch_datasets.json --dry_run
python run_batch.py --config batch_datasets.json
python run_batch.py --config batch_datasets.json --datasets ACM,DBLP
```

如果只需要复现原始 EDA-GCL 基线，可运行：

```bash
sh script/train_homo.sh
sh script/train_hete.sh
```

Windows 环境没有原生 `sh` 时，可以直接按照脚本中的参数运行对应的 Python 文件，或使用 Git Bash/WSL。

## 训练输出

训练过程会输出节点表示学习和社区搜索结果，常见指标包括：

- 节点分类 Micro-F1、Macro-F1；
- 社区搜索 Precision、Recall、F1 和 Jaccard；
- 可疑节点 Precision@K、Recall@K；
- 被删边恢复和伪装边识别的 AUC/AP；
- 查询意图与搜索社区的相关性。

训练日志默认保存在 `log/` 或批量实验生成的输出目录中，模型检查点保存在 `checkpoints/`。

## 方法设计文档

项目中的方法说明和实验背景主要位于：

- `article/创新点摘要.md`
- `article/意图向量获取方法.md`
- `article/对抗模式库资源汇总.md`
- `article/Edge Self-Adversarial Augmentation Enhances Graph Contrastive Learning.pdf`

## 原始工作

本项目的图对比学习基线来源于 EDA-GCL：

> C. Chen et al., “Edge Self-Adversarial Augmentation Enhances Graph Contrastive Learning Against Neighborhood Inconsistency,” Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

```bibtex
@inproceedings{chen2026edge,
  title={Edge Self-Adversarial Augmentation Enhances Graph Contrastive Learning Against Neighborhood Inconsistency},
  author={Chen, Chunchun and Wei, Xing and Yang, Jiayi and Wang, Chenrun and Fu, Yiwei and Zhang, Yuxing and Sun, Xin and Fan, Rui and Ye, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={24},
  pages={20005--20013},
  year={2026}
}
```

## 联系方式

- Email: binzhaos@163.com
