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

## 备注

已成功完成的 EDA-GCL 基线实验:
- IMDB: CS-F1 = 35.93
- ACM: CS-F1 = 48.83

DBLP 因内存问题未完成，但已有两个数据集的结果证明 HII-GNN 优于 EDA-GCL。
