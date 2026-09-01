from data_loader import get_dataset, get_dataset1

import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args
from model import PretrainModel
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
from utils import *



if __name__ == "__main__":
    # 解析输入参数
    args = parse_args()
    print(args)

    # 设置随机种子，保证实验可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 加载数据集
    # adj: (N, N), features: (N, d(加过位置编码维度))
    # adj, features = get_dataset(args.dataset, args.pe_dim)
    adj, features = get_dataset1(args.dataset, args.pe_dim)

    # 特征预处理
    start_feature_processing = time.time()
    # 处理后的特征 shape: (N, hops+1, d)
    processed_features = utils.re_features(adj, features, args.hops)  # return (N, hops+1, d)
    if processed_features.shape[0] < 10000:         # 节点数小于10000
        # 指标矩阵 shape: (N, hops+1)
        indicator = utils.conductance_hop(adj, args.hops) # return (N, hops+1)
        # 将指标矩阵扩展维度为 shape: (N, hops+1, d)
        indicator = indicator.unsqueeze(2).repeat(1, 1, features.shape[1])
        # 特征矩阵按指标矩阵进行加权
        processed_features = processed_features*indicator
    t_feature_precessing = time.time() - start_feature_processing
    print("特征处理时间: {:.4f}s".format(t_feature_precessing))

    # 邻接矩阵处理
    start = time.time()
    print("开始转换为 COO 格式")

    # 将邻接矩阵从 COO 格式转换为 CSR 格式以便高效切片
    adj = transform_coo_to_csr(adj)  # 转换为 CSR 格式以支持切片操作
    print("开始小批量处理")    # mini batch

    # 将邻接矩阵分块为小批量处理的格式
    # 返回两个列表，`adj_batch` 和 `minus_adj_batch`
    # 分别是邻接矩阵和“反邻接矩阵”的小批次版本
    adj_batch, minus_adj_batch = transform_sp_csr_to_coo(adj, args.batch_size, features.shape[0]) # transform to coo to support tensor operation
    # adj_batch: 每批次的邻接矩阵（list of (batch_size, batch_size) in COO 格式）
    # minus_adj_batch: 每批次的减法邻接矩阵（list of (batch_size, batch_size) in COO 格式）
    print(len(adj_batch[0]), len(minus_adj_batch[0]))
    print("邻接矩阵处理时间: {:.4f}s".format(time.time() - start))

    # 创建数据加载器，用于小批量加载节点特征
    data_loader = Data.DataLoader(processed_features, batch_size=args.batch_size, shuffle = False)

    # model configuration  初始化模型
    model = PretrainModel(input_dim=processed_features.shape[2], config=args).to(args.device)
    print(model)
    print('总参数数量:', sum(p.numel() for p in model.parameters()))

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    # 定义学习率调度器
    lr_scheduler = PolynomialDecayLR(
                    optimizer,
                    warmup_updates=args.warmup_updates,
                    tot_updates=args.tot_updates,
                    lr=args.peak_lr,
                    end_lr=args.end_lr,
                    power=1.0)

    # 定义早停策略
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)

    print("开始训练...")
    # 模型训练
    model.train()
    t_start = time.time()
    loss_train_b = []

    for epoch in range(args.epochs):  # 遍历所有的训练轮次
        for index, item in enumerate(data_loader):  # 遍历每个批次的数据

            # 获取当前批次的起始索引
            start_index = index * args.batch_size

            # 加载当前批次的节点特征
            nodes_features = item.to(args.device)

            # 加载对应的邻接矩阵
            adj_ = adj_batch[index].to(args.device)
            minus_adj = minus_adj_batch[index].to(args.device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            node_tensor, neighbor_tensor = model(nodes_features)

            # 计算对比损失
            loss_train = model.contrastive_link_loss(node_tensor, neighbor_tensor, adj_, minus_adj)

            # 反向传播
            loss_train.backward()
            optimizer.step()
            lr_scheduler.step()

            # 记录损失值
            loss_train_b.append(loss_train.item())

        # 检查早停条件
        if early_stopping.simple_check(loss_train_b):
            break

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()))

        # 'loss_train: {:.4f}'.format(np.mean(np.array(loss_train_b)))
    
    print("优化完成!")
    print("训练时间: {:.4f}s".format(time.time() - t_start + t_feature_precessing))

    # model save    模型保存
    print("开始保存模型...")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path)

    # 保存模型权重
    torch.save(model.state_dict(), args.save_path + args.model_name + '.pth')
    
    # obtain all the node embedding from the learned model
    # 提取节点嵌入
    model.eval()
    node_embedding = []
    for _, item in enumerate(data_loader):      # 遍历所有节点特征
        nodes_features = item.to(args.device)
        node_tensor, neighbor_tensor = model(nodes_features)
        # 将当前批次的节点嵌入保存
        if len(node_embedding) == 0:
            node_embedding = np.concatenate((node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
            # node_embedding = node_tensor.cpu().detach().numpy()
        else:
            new_node_embedding = np.concatenate((node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
            # new_node_embedding = node_tensor.cpu().detach().numpy()
            node_embedding = np.concatenate((node_embedding, new_node_embedding), axis=0)

    # 保存节点嵌入
    np.save(args.embedding_path + args.model_name + '.npy', node_embedding)

    




def test(model, config, train_idx_list, val_idx_list, test_idx_list, labels, num_classes, fea_evalue, ma_dic_list,
         mi_dic_list, auc_dic_list, dgl_new_het_graph, category):
    """
    评估训练模型的性能，测试其在验证集和测试集上的表现，并计算分类的评估指标（如准确率、AUC等）。

    :param model: 需要评估的训练模型。
    :param config: 配置参数，包含数据集和模型设置。
    :param train_idx_list: 不同数据划分中训练集节点的索引列表。
    :param val_idx_list: 不同数据划分中验证集节点的索引列表。
    :param test_idx_list: 不同数据划分中测试集节点的索引列表。
    :param labels: 节点的真实标签。
    :param num_classes: 分类任务中的类别数。
    :param fea_evalue: 用于评估的特征嵌入（通常是多跳特征）。
    :param ma_dic_list: 存储不同数据划分下的平均准确率（ma）的字典。
    :param mi_dic_list: 存储不同数据划分下的平均逆（mi）的字典。
    :param auc_dic_list: 存储不同数据划分下的AUC（曲线下面积）的字典。
    :return: 无返回值，结果保存在字典中。
    """
    # 记录评估开始时间
    starttime = datetime.datetime.now()
    model.eval()  # 设置模型为评估模式（关闭Dropout等训练时的操作）

    # 获取模型的嵌入结果，这里 fea_evalue.permute(1, 0, 2, 3) 可能表示特征的重排列，具体依赖于模型的设计
    emb = model.get_embeds(multi_hop_features=fea_evalue.permute(1, 0, 2, 3))  # (batch_size, feature_dim)

    print(f"训练后的嵌入形状: {emb.shape}")

    # 获取嵌入特征矩阵并将其保存到文件
    emb_tensor = emb.detach().cpu().numpy()  # 转换为numpy数组
    emb_filename = "../data/" + args.dataset + "Pretrained_node_embedding.npy"  # 存储文件名
    np.save(emb_filename, emb_tensor)  # 保存到.npy文件

    # 确保文件保存成功
    if os.path.exists(emb_filename):
        print(f"节点嵌入保存成果 {emb_filename}")
    else:
        print(f"节点嵌入保存失败 {emb_filename}")

    # 读取保存的嵌入文件
    loaded_emb = np.load(emb_filename)  # 从.npy文件读取嵌入矩阵
    print(f"加载嵌入成果，加载的嵌入的形状: {loaded_emb.shape}")

    # 将加载的嵌入矩阵转换为torch tensor
    emb_tensor_loaded = torch.from_numpy(loaded_emb).float()  # (batch_size, feature_dim)





