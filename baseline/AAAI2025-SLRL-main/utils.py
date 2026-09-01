import os
import sys

import numpy as np
import torch
import random

# 添加共享模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.dataset_utils import get_dataset_list, HETE_DATASETS


def seed_all(seed):
    '''
    固定随机种子
    @param seed: 种子
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def getFileInfo(type):
    '''
    获取种子节点或真实社区下标
    @param type: 文件名
    @return: 二维列表，所有数据集对应的种子节点或者真实社区下标
    '''
    seed = []
    filename = f'datasets/{type}'
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            line = list(map(int, line))
            seed.append(line)
    return seed

def getseedsAndtruecom(args, dataset):
    '''
    获取当前数据集选取的种子节点以及对应真实社区的下标
    @param dataset: 数据集名称
    @param search_size: 检测数量
    @param start: 开始位置
    '''
    # 数据集名称映射（支持别名）
    name_map = {
        'imdb_new': 'imdb_new',
        'imdb': 'imdb_new',
        'acm': 'acm',
        'dblp': 'dblp',
        'amazon': 'amazon',
        'lj': 'lj',
        'youtube': 'youtube',
        'twitter': 'twitter',
        'facebook': 'facebook'
    }

    dataset_lower = dataset.lower()
    if dataset_lower in name_map:
        dataset_lower = name_map[dataset_lower]

    # 直接按数据集名称查找索引
    dataset_order = ['acm', 'dblp', 'imdb_new']
    if dataset_lower not in dataset_order:
        raise ValueError(f"Dataset {dataset} not supported. Available: {dataset_order}")

    idx = dataset_order.index(dataset_lower)

    seeds = getFileInfo(f"seed12")[idx]
    com_indexs = getFileInfo(f"com_index12")[idx]
    return seeds, com_indexs

def writerResToFile(args, res):
    '''
    将结果写进文件种
    @param args:
    @param res: res[0]保存种子节点，res[1]保存对应真实社区下标，res[2]表示SLRL生成的社区
    @return:
    '''
    seed, comindex, coms = res[0], res[1], res[2]

    # rootDir = f"./AAAi/{args.resfileName}"
    # if args.way == "qian":
    #     rootDir = f"./k{args.k}/{args.resfileName}"
    rootDir = f"./res"

    with open(f'{rootDir}/{args.dataset}_seed.txt', 'a') as file:
        file.write(f"{seed} ")
    with open(f'{rootDir}/{args.dataset}_com_index.txt', 'a') as file:
        file.write(f"{comindex} ")
    with open(f'{rootDir}/{args.dataset}_pred_com.txt', 'a') as file:
        for node in coms:
            file.write(f"{node} ")
        file.write("\n")


def wr_file(seed, comindex, pred_com, args):

    rootDir = f"./AAAi/com_all"
    with open(f'{rootDir}/{args.dataset}_seed.txt', 'a') as file:
        file.write(f"{seed} ")
    with open(f'{rootDir}/{args.dataset}_com_index.txt', 'a') as file:
        file.write(f"{comindex} ")
    with open(f'{rootDir}/{args.dataset}_pred_com.txt', 'a') as file:
        for node in pred_com:
            file.write(f"{node} ")
        file.write("\n")

