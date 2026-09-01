#!/bin/bash
# 测试改进版社区搜索的三种方法

export KMP_DUPLICATE_LIB_OK=TRUE

DATASET="IMDB"
TOPK_RATIO=0.2
NUM_QUERIES=150

echo "=========================================="
echo "数据集: $DATASET"
echo "Top-k比例: $TOPK_RATIO"
echo "查询数: $NUM_QUERIES"
echo "=========================================="
echo ""

# 方法1: 图感知 (BFS + 相似度)
echo "【方法1】图感知社区搜索 (graph-aware)"
echo "------------------------------------------"
python community_search_improved.py \
    --dataset $DATASET \
    --method graph \
    --topk_ratio $TOPK_RATIO \
    --num_queries $NUM_QUERIES \
    --hop 2
echo ""

# 方法2: KNN扩展
echo "【方法2】KNN扩展社区搜索"
echo "------------------------------------------"
python community_search_improved.py \
    --dataset $DATASET \
    --method knn \
    --topk_ratio $TOPK_RATIO \
    --num_queries $NUM_QUERIES \
    --k 10
echo ""

# 方法3: 密度感知
echo "【方法3】密度感知社区搜索"
echo "------------------------------------------"
python community_search_improved.py \
    --dataset $DATASET \
    --method density \
    --topk_ratio $TOPK_RATIO \
    --num_queries $NUM_QUERIES \
    --threshold 0.5
echo ""

echo "=========================================="
echo "所有测试完成!"
echo "=========================================="
