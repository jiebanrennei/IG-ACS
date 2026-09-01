#!/bin/bash
# 快速验证脚本：测试异构 + 同质数据集

echo "============================================================"
echo "Quick Validation: Heterogeneous + Homogeneous"
echo "============================================================"
echo ""

# 测试异构图（小参数快速验证）
echo "[Test 1/4] ACM (heterogeneous) - quick test..."
python train_ig.py --dataset ACM --encoder hii --num_hidden 128 --num_epochs 10 --sparsify_topk 50
echo ""

# 测试同质图（从小数据集开始）
echo "[Test 2/4] com-amazon (homogeneous) - quick test..."
python train_ig.py --dataset com-amazon --encoder gcn --num_hidden 128 --num_epochs 10
echo ""

echo "[Test 3/4] com-dblp (homogeneous) - quick test..."
python train_ig.py --dataset com-dblp --encoder gcn --num_hidden 128 --num_epochs 10
echo ""

echo "[Test 4/4] com-youtube (homogeneous) - quick test..."
python train_ig.py --dataset com-youtube --encoder gcn --num_hidden 128 --num_epochs 10
echo ""

echo "============================================================"
echo "Validation completed!"
echo "============================================================"
