#!/bin/bash
# IG-ACS 完整实验脚本：异构 + 同质数据集

echo "============================================================"
echo "IG-ACS Complete Experiment Script"
echo "============================================================"
echo ""

# 通用参数
NUM_HIDDEN=256
NUM_EPOCHS=200
ENCODER="gcn"

# ============================================================
# 第一部分：异构图数据集 (Heterogeneous Graphs)
# ============================================================
echo "============================================================"
echo "Part 1: Heterogeneous Graph Datasets"
echo "============================================================"
echo ""

echo "[1/3] Running ACM (heterogeneous)..."
python train_ig.py --dataset ACM --encoder hii --num_hidden $NUM_HIDDEN --num_epochs $NUM_EPOCHS --sparsify_topk 50 > log/acm_hii.log 2>&1
echo "  Done! Check log/acm_hii.log"
echo ""

echo "[2/3] Running DBLP (heterogeneous)..."
python train_ig.py --dataset DBLP --encoder gcn --base_model GATConv --num_hidden $NUM_HIDDEN --num_epochs $NUM_EPOCHS > log/dblp_gat.log 2>&1
echo "  Done! Check log/dblp_gat.log"
echo ""

echo "[3/3] Running IMDB_NEW (heterogeneous)..."
python train_ig.py --dataset IMDB_NEW --encoder hii --num_hidden $NUM_HIDDEN --num_epochs $NUM_EPOCHS > log/imdb_hii.log 2>&1
echo "  Done! Check log/imdb_hii.log"
echo ""

# ============================================================
# 第二部分：同质图数据集 (Homogeneous Graphs) - SLRL 对比
# ============================================================
echo "============================================================"
echo "Part 2: Homogeneous Graph Datasets (for SLRL comparison)"
echo "============================================================"
echo ""

echo "[1/5] Running com-amazon (homogeneous)..."
python train_ig.py --dataset com-amazon --encoder $ENCODER --num_hidden $NUM_HIDDEN --num_epochs $NUM_EPOCHS > log/com_amazon.log 2>&1
echo "  Done! Check log/com_amazon.log"
echo ""

echo "[2/5] Running com-dblp (homogeneous)..."
python train_ig.py --dataset com-dblp --encoder $ENCODER --num_hidden $NUM_HIDDEN --num_epochs $NUM_EPOCHS > log/com_dblp.log 2>&1
echo "  Done! Check log/com_dblp.log"
echo ""

echo "[3/5] Running com-youtube (homogeneous)..."
python train_ig.py --dataset com-youtube --encoder $ENCODER --num_hidden $NUM_HIDDEN --num_epochs $NUM_EPOCHS > log/com_youtube.log 2>&1
echo "  Done! Check log/com_youtube.log"
echo ""

echo "[4/5] Running com-twitter (homogeneous)..."
python train_ig.py --dataset com-twitter --encoder $ENCODER --num_hidden $NUM_HIDDEN --num_epochs $NUM_EPOCHS > log/com_twitter.log 2>&1
echo "  Done! Check log/com_twitter.log"
echo ""

echo "[5/5] Running com-livejournal (homogeneous)..."
python train_ig.py --dataset com-livejournal --encoder $ENCODER --num_hidden $NUM_HIDDEN --num_epochs $NUM_EPOCHS > log/com_lj.log 2>&1
echo "  Done! Check log/com_lj.log"
echo ""

echo "============================================================"
echo "All experiments completed!"
echo "============================================================"
echo ""
echo "Results summary:"
echo "  - Heterogeneous: ACM, DBLP, IMDB_NEW"
echo "  - Homogeneous: com-amazon, com-dblp, com-youtube, com-twitter, com-livejournal"
echo ""
echo "Next steps:"
echo "  1. Check logs in log/ directory"
echo "  2. Run SLRL on the same homogeneous datasets for comparison"
echo "  3. Update experiment_log.md with results"
