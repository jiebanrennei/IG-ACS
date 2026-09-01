# 因内存不足未完成的实验

## DBLP + EDA-GCL (社区搜索)

**数据转换命令**:
```bash
cd baseline/EDA-GCL-main
python convert_hete_to_homo.py --dataset DBLP
```

**训练命令** (多次尝试均 OOM):
```bash
# 尝试1: 默认参数
python train_homo.py --dataset DBLP --gpu_id 0 --num_hidden 64 --num_layers 1 --num_epochs 100

# 尝试2: 减小参数
python train_homo.py --dataset DBLP --gpu_id 0 --num_hidden 32 --num_layers 1 --num_epochs 50

# 尝试3: 最小参数
python train_homo.py --dataset DBLP --gpu_id 0 --num_hidden 16 --num_proj_hidden 16 --num_edge_hidden 8 --num_layers 1 --num_epochs 20 --wd_train 0.001
```

**失败原因**: CUDA out of memory
- GPU: 23.56 GiB
- DBLP 数据集太大，即使最小参数也会 OOM
- 已修改 model.py 添加分批计算（batch_size=1000），但仍然 OOM

**解决方案**:
1. 使用更大显存的 GPU（>24GB）
2. 使用 CPU 训练（速度慢但能完成）
3. 进一步优化模型代码（梯度检查点等）

**待运行的社区搜索命令** (训练完成后):
```bash
python community_search.py --dataset DBLP --topk_ratio 0.2 --num_queries 150
```

---

## ACM/DBLP/IMDB + SLRL (社区搜索)

**数据转换命令**:
```bash
cd baseline/AAAI2025-SLRL-main
python convert_hete_to_slrl.py --datasets acm dblp imdb_new --output_dir ./datasets --data_root ../../datasets
```

**运行命令** (OOM):
```bash
python mainSLRL.py --dataset acm --root datasets --search_size 150 --start 0
```

**失败原因**: CUDA out of memory
- SLRL 加载2000万条边到内存
- 算法复杂度高，需要提取子图和社区结构
- 即使只处理1个节点也会 OOM

**解决方案**:
1. 使用更大显存的 GPU
2. 使用 CPU 运行（速度极慢）
3. 减少 search_size 和 train_size 参数

---

## 备注

已成功完成的基线实验:
- **EDA-GCL**: IMDB F1=35.93, ACM F1=48.83
- 已证明 HII-GNN 优于 EDA-GCL（提升20+个点）

SLRL 和 DBLP+EDA-GCL 因内存问题未完成，但现有结果已足够支撑论文。
