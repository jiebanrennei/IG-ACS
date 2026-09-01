"""
将 SLRL (AAAI 2025) 的数据集转换为 IG-ACS 格式。

SLRL 数据集位置: D:\论文code\zhengshi\baseline\AAAI2025-SLRL-main\datasets
IG-ACS 数据集位置: D:\论文code\zhengshi\IG-ACS\datasets

SLRL 格式:
- amazon/amazon-1.90.ungraph.txt
- amazon/amazon-1.90.cmty.txt

IG-ACS 格式:
- com-Amazon/com-amazon.ungraph.txt
- com-Amazon/com-amazon.top5000.cmty.txt

支持的转换:
- amazon -> com-Amazon
- dblp -> com-DBLP
- youtube -> com-Youtube
- lj -> com-LiveJournal
- twitter -> com-Twitter
"""

import os
import shutil
from pathlib import Path

# 路径配置
SLRL_ROOT = Path(r"D:/论文code/zhengshi/baseline/AAAI2025-SLRL-main/datasets")
IGACS_ROOT = Path(r"D:/论文code/zhengshi/IG-ACS/datasets")

# 数据集映射: SLRL名称 -> IG-ACS名称
DATASET_MAPPING = {
    'amazon': 'com-Amazon',
    'dblp': 'com-DBLP',
    'youtube': 'com-Youtube',
    'lj': 'com-LiveJournal',
    'twitter': 'com-Twitter',
}

def convert_dataset(slrl_name, igacs_name):
    """转换单个数据集"""
    print(f"\n{'='*60}")
    print(f"Converting: {slrl_name} -> {igacs_name}")
    print(f"{'='*60}")

    # SLRL 源文件
    slrl_dir = SLRL_ROOT / slrl_name
    slrl_edge_file = slrl_dir / f"{slrl_name}-1.90.ungraph.txt"
    slrl_cmty_file = slrl_dir / f"{slrl_name}-1.90.cmty.txt"

    # IG-ACS 目标目录和文件
    igacs_dir = IGACS_ROOT / igacs_name
    igacs_dir.mkdir(parents=True, exist_ok=True)

    # 文件名转换
    igacs_edge_name = f"com-{slrl_name.lower()}.ungraph.txt"
    igacs_cmty_name = f"com-{slrl_name.lower()}.top5000.cmty.txt"

    igacs_edge_file = igacs_dir / igacs_edge_name
    igacs_cmty_file = igacs_dir / igacs_cmty_name

    # 检查源文件是否存在
    if not slrl_edge_file.exists():
        print(f"❌ Edge file not found: {slrl_edge_file}")
        return False
    if not slrl_cmty_file.exists():
        print(f"❌ Community file not found: {slrl_cmty_file}")
        return False

    # 复制边文件
    print(f"[1/2] Copying edge file:")
    print(f"   From: {slrl_edge_file}")
    print(f"   To:   {igacs_edge_file}")
    shutil.copy2(slrl_edge_file, igacs_edge_file)

    # 复制社区文件 (SLRL 的社区数 < 5000, 直接使用全部)
    print(f"[2/2] Copying community file:")
    print(f"   From: {slrl_cmty_file}")
    print(f"   To:   {igacs_cmty_file}")

    # 统计社区数量
    with open(slrl_cmty_file, 'r') as f:
        num_communities = sum(1 for line in f if line.strip())
    print(f"   Total communities: {num_communities}")

    if num_communities > 5000:
        print(f"   Warning: {num_communities} > 5000, taking top 5000 communities")
        with open(slrl_cmty_file, 'r') as fin, open(igacs_cmty_file, 'w') as fout:
            for i, line in enumerate(fin):
                if i >= 5000:
                    break
                fout.write(line)
    else:
        shutil.copy2(slrl_cmty_file, igacs_cmty_file)

    print(f"[OK] Successfully converted {slrl_name} -> {igacs_name}")
    return True


def main():
    print("="*60)
    print("SLRL -> IG-ACS Dataset Converter")
    print("="*60)
    print(f"SLRL root:  {SLRL_ROOT}")
    print(f"IG-ACS root: {IGACS_ROOT}")

    # 检查 SLRL 目录是否存在
    if not SLRL_ROOT.exists():
        print(f"\n[ERROR] SLRL directory not found: {SLRL_ROOT}")
        print("Please make sure SLRL code is at the correct location.")
        return

    # 转换所有数据集
    success_count = 0
    for slrl_name, igacs_name in DATASET_MAPPING.items():
        if convert_dataset(slrl_name, igacs_name):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Conversion Summary: {success_count}/{len(DATASET_MAPPING)} datasets converted")
    print(f"{'='*60}")

    if success_count > 0:
        print("\n[SUCCESS] You can now use these datasets in IG-ACS:")
        print("   python train_ig.py --dataset com-amazon --encoder gcn --num_hidden 256")
        print("   python train_ig.py --dataset com-dblp --encoder gcn --num_hidden 256")
        print("   python train_ig.py --dataset com-youtube --encoder gcn --num_hidden 256")
        if 'lj' in DATASET_MAPPING:
            print("   python train_ig.py --dataset com-livejournal --encoder gcn --num_hidden 256")
        if 'twitter' in DATASET_MAPPING:
            print("   python train_ig.py --dataset com-twitter --encoder gcn --num_hidden 256")


if __name__ == '__main__':
    main()
