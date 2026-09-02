#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集信息统计工具（异构图 + 同构图）
生成数据集的详细统计表格
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from pathlib import Path
from scipy import sparse


# 异构图数据集配置
HETE_DATASETS = {
    'ACM': {
        'dir': 'acm',
        'node_types': ['paper', 'author', 'subject'],
        'meta_paths': {
            'PAP': 'pap.npz',
            'PSP': 'psp.npz'
        },
        'edges': {
            'P-A': 'pa.txt',
            'P-S': 'ps.txt'
        }
    },
    'DBLP': {
        'dir': 'dblp',
        'node_types': ['author', 'paper', 'term', 'conference'],
        'meta_paths': {
            'APA': 'apa.npz',
            'APCPA': 'apcpa.npz',
            'APTPA': 'aptpa.npz'
        },
        'edges': {
            'P-A': 'pa.txt',
            'P-T': 'pt.txt',
            'P-C': 'pc.txt'
        }
    },
    'IMDB': {
        'dir': 'imdb_new',
        'node_types': ['movie', 'actor', 'director'],
        'meta_paths': {
            'MAM': 'mam.npz',
            'MDM': 'mdm.npz'
        },
        'edges': {
            'M-D': 'md.txt',
            'M-A': 'ma.txt'
        }
    }
}

# 同构图数据集配置
HOMO_DATASETS = {
    'amazon': {
        'dir': 'amazon',
        'info_file': 'amazon.90.info.txt',
        'graph_file': 'amazon-1.90.ungraph.txt',
        'community_file': None
    },
    'com-amazon': {
        'dir': 'com-Amazon',
        'info_file': None,
        'graph_file': 'com-amazon.ungraph.txt',
        'community_file': 'com-amazon.top5000.cmty.txt'
    },
    'com-dblp': {
        'dir': 'com-DBLP',
        'info_file': None,
        'graph_file': 'com-dblp.ungraph.txt',
        'community_file': 'com-dblp.top5000.cmty.txt'
    },
    'com-youtube': {
        'dir': 'com-Youtube',
        'info_file': None,
        'graph_file': 'com-youtube.ungraph.txt',
        'community_file': 'com-youtube.top5000.cmty.txt'
    },
    'com-twitter': {
        'dir': 'com-Twitter',
        'info_file': None,
        'graph_file': 'com-twitter.ungraph.txt',
        'community_file': 'com-twitter.top5000.cmty.txt'
    },
    'com-lj': {
        'dir': 'com-LiveJournal',
        'info_file': None,
        'graph_file': 'com-lj.ungraph.txt',
        'community_file': 'com-lj.top5000.cmty.txt'
    }
}


def count_nodes_from_features(filepath):
    """从特征文件统计节点数量"""
    if not os.path.exists(filepath):
        return 0
    try:
        # 尝试作为稀疏矩阵加载
        data = sparse.load_npz(filepath)
        return data.shape[0]
    except:
        try:
            # 尝试作为普通 numpy 数组加载
            data = np.load(filepath)
            if hasattr(data, 'shape'):
                return data.shape[0]
        except:
            pass
    return 0


def count_edges_from_txt(filepath):
    """从 txt 文件统计边数量"""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return len(lines)


def parse_homo_info_file(filepath):
    """解析同构图 info 文件"""
    if not os.path.exists(filepath):
        return None, None

    num_nodes = None
    num_edges = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '# nodes =' in line:
                num_nodes = int(line.split('=')[1].strip().replace(',', ''))
            elif '# edges =' in line:
                num_edges = int(line.split('=')[1].strip().replace(',', ''))

    return num_nodes, num_edges


def count_edges_from_graph_file(filepath):
    """从图文件统计边数量"""
    if not os.path.exists(filepath):
        return 0

    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                count += 1
    return count


def analyze_community_file(filepath):
    """分析社区文件，返回社区统计信息"""
    if not os.path.exists(filepath):
        return None

    communities = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                nodes = [int(x) for x in line.strip().split()]
                communities.append(nodes)

    if not communities:
        return None

    sizes = [len(c) for c in communities]

    stats = {
        'num_communities': len(communities),
        'min_size': min(sizes),
        'max_size': max(sizes),
        'avg_size': np.mean(sizes),
        'std_size': np.std(sizes),
        'median_size': np.median(sizes),
    }

    return stats


def analyze_hete_dataset(dataset_name, data_root='./datasets'):
    """分析异构图数据集"""
    dataset_name = dataset_name.upper()

    if dataset_name not in HETE_DATASETS:
        raise ValueError(f"未知异构图数据集: {dataset_name}")

    cfg = HETE_DATASETS[dataset_name]
    data_dir = os.path.join(data_root, cfg['dir'])

    info = {
        'name': dataset_name,
        'type': 'heterogeneous',
        'node_types': {},
        'edge_types': {},
        'meta_paths': list(cfg['meta_paths'].keys()),
        'avg_degree': 0.0
    }

    # 统计节点数量
    features_map = {
        'ACM': {'paper': 'p_feat.npz', 'author': 'a_feat.npz'},
        'DBLP': {'author': 'a_feat.npz', 'paper': 'p_feat.npz', 'term': 't_feat.npz'},
        'IMDB': {'movie': 'm_feat.npz'}
    }

    if dataset_name in features_map:
        for ntype, feat_file in features_map[dataset_name].items():
            feat_path = os.path.join(data_dir, feat_file)
            count = count_nodes_from_features(feat_path)
            if count > 0:
                info['node_types'][ntype] = count

    # 从 labels 获取节点数
    labels_path = os.path.join(data_dir, 'labels.npy')
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        if dataset_name == 'ACM':
            info['node_types']['paper'] = len(labels)
        elif dataset_name == 'DBLP':
            info['node_types']['author'] = len(labels)
        elif dataset_name == 'IMDB':
            info['node_types']['movie'] = len(labels)

    # 从 node_types.npy 获取节点类型分布
    node_types_path = os.path.join(data_dir, 'node_types.npy')
    if os.path.exists(node_types_path):
        try:
            node_types = np.load(node_types_path, allow_pickle=True)
            if node_types.dtype == object:
                unique, counts = np.unique(node_types, return_counts=True)
                for ntype, count in zip(unique, counts):
                    info['node_types'][str(ntype)] = count
            else:
                unique, counts = np.unique(node_types, return_counts=True)
                type_names = cfg['node_types']
                for i, count in zip(unique, counts):
                    if i < len(type_names):
                        info['node_types'][type_names[i]] = count
        except:
            pass

    # 统计边数量
    for edge_name, edge_file in cfg['edges'].items():
        edge_path = os.path.join(data_dir, edge_file)
        count = count_edges_from_txt(edge_path)
        if count > 0:
            info['edge_types'][edge_name] = count

    # 计算平均度（2*|E|/|V|）
    total_edges = sum(info['edge_types'].values())
    total_nodes = sum(info['node_types'].values())
    if total_nodes > 0:
        info['avg_degree'] = 2 * total_edges / total_nodes

    return info


def analyze_homo_dataset(dataset_name, data_root='./datasets'):
    """分析同构图数据集"""
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower not in HOMO_DATASETS:
        raise ValueError(f"未知同构图数据集: {dataset_name}")

    cfg = HOMO_DATASETS[dataset_name_lower]
    data_dir = os.path.join(data_root, cfg['dir'])

    info = {
        'name': dataset_name,
        'type': 'homogeneous',
        'num_nodes': 0,
        'num_edges': 0,
        'avg_degree': 0.0,
        'community_stats': None
    }

    # 尝试从 info 文件读取
    if cfg['info_file']:
        info_path = os.path.join(data_dir, cfg['info_file'])
        num_nodes, num_edges = parse_homo_info_file(info_path)
        if num_nodes:
            info['num_nodes'] = num_nodes
        if num_edges:
            info['num_edges'] = num_edges

    # 如果 info 文件没有提供，从图文件统计
    if info['num_nodes'] == 0 or info['num_edges'] == 0:
        graph_path = os.path.join(data_dir, cfg['graph_file'])
        if os.path.exists(graph_path):
            # 统计边数
            if info['num_edges'] == 0:
                info['num_edges'] = count_edges_from_graph_file(graph_path)

            # 统计节点数（从边列表中找最大节点 ID）
            if info['num_nodes'] == 0:
                max_node = 0
                with open(graph_path, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    u, v = int(parts[0]), int(parts[1])
                                    max_node = max(max_node, u, v)
                                except:
                                    pass
                info['num_nodes'] = max_node + 1

    # 计算平均度
    if info['num_nodes'] > 0:
        info['avg_degree'] = 2 * info['num_edges'] / info['num_nodes']

    # 分析社区文件
    if cfg.get('community_file'):
        community_path = os.path.join(data_dir, cfg['community_file'])
        community_stats = analyze_community_file(community_path)
        if community_stats:
            info['community_stats'] = community_stats

    return info


def print_hete_summary(datasets, data_root='./datasets'):
    """打印异构图统计摘要"""
    print("\n" + "="*120)
    print("异构图数据集统计摘要")
    print("="*120 + "\n")

    print(f"{'数据集':<12} {'节点数量':<45} {'边的数量':<45} {'元路径':<20} {'平均度':<8}")
    print("-"*120)

    for dataset_name in datasets:
        try:
            info = analyze_hete_dataset(dataset_name, data_root)

            node_parts = []
            for ntype, count in info['node_types'].items():
                abbr = ntype[0].upper()
                node_parts.append(f"{ntype}({abbr}):{count}")
            node_str = "\n".join(node_parts) if node_parts else "-"

            edge_parts = []
            for etype, count in info['edge_types'].items():
                edge_parts.append(f"{etype}:{count}")
            edge_str = "\n".join(edge_parts) if edge_parts else "-"

            mp_str = "\n".join(info['meta_paths']) if info['meta_paths'] else "-"
            degree_str = f"{info['avg_degree']:.1f}"

            node_lines = node_str.split('\n')
            edge_lines = edge_str.split('\n')
            mp_lines = mp_str.split('\n')
            max_lines = max(len(node_lines), len(edge_lines), len(mp_lines), 1)

            for i in range(max_lines):
                if i == 0:
                    print(f"{info['name']:<12}", end='')
                else:
                    print(f"{'':<12}", end='')

                node_part = node_lines[i] if i < len(node_lines) else ''
                edge_part = edge_lines[i] if i < len(edge_lines) else ''
                mp_part = mp_lines[i] if i < len(mp_lines) else ''

                print(f" {node_part:<43} {edge_part:<43} {mp_part:<18} {degree_str:<8}")

            print("-"*120)
        except Exception as e:
            print(f"{dataset_name:<12} [错误] {str(e)}")
            print("-"*120)


def print_homo_summary(datasets, data_root='./datasets'):
    """打印同构图统计摘要"""
    print("\n" + "="*100)
    print("同构图数据集统计摘要")
    print("="*100 + "\n")

    print(f"{'数据集':<15} {'节点数':<10} {'边数':<10} {'平均度':<8} {'社区数':<8} {'社区大小(min/avg/max)':<25}")
    print("-"*100)

    for dataset_name in datasets:
        try:
            info = analyze_homo_dataset(dataset_name, data_root)

            # 社区信息
            if info['community_stats']:
                cs = info['community_stats']
                num_comm = cs['num_communities']
                size_str = f"{cs['min_size']}/{cs['avg_size']:.1f}/{cs['max_size']}"
            else:
                num_comm = "-"
                size_str = "-"

            print(f"{info['name']:<15} {info['num_nodes']:<10,} {info['num_edges']:<10,} "
                  f"{info['avg_degree']:<8.2f} {num_comm:<8} {size_str:<25}")
        except Exception as e:
            print(f"{dataset_name:<15} [错误] {str(e)}")

    print("-"*100)


def generate_markdown_table(hete_datasets, homo_datasets, data_root='./datasets', output_file=None):
    """生成 Markdown 格式的表格"""
    lines = []
    lines.append("# 数据集统计摘要\n")

    # 异构图表格
    lines.append("## 异构图数据集\n")
    lines.append("| 数据集 | 节点数量 | 边的数量 | 元路径 | 平均度 |")
    lines.append("|--------|----------|----------|--------|--------|")

    for dataset_name in hete_datasets:
        try:
            info = analyze_hete_dataset(dataset_name, data_root)

            node_parts = []
            for ntype, count in info['node_types'].items():
                abbr = ntype[0].upper()
                node_parts.append(f"{ntype}({abbr}):{count}")
            node_str = "<br>".join(node_parts) if node_parts else "-"

            edge_parts = []
            for etype, count in info['edge_types'].items():
                edge_parts.append(f"{etype}:{count}")
            edge_str = "<br>".join(edge_parts) if edge_parts else "-"

            mp_str = "<br>".join(info['meta_paths']) if info['meta_paths'] else "-"
            degree_str = f"{info['avg_degree']:.1f}"

            lines.append(f"| {info['name']} | {node_str} | {edge_str} | {mp_str} | {degree_str} |")
        except Exception as e:
            lines.append(f"| {dataset_name} | [错误] {str(e)} | - | - | - |")

    # 同构图表格
    lines.append("\n## 同构图数据集\n")
    lines.append("| 数据集 | 节点数 | 边数 | 平均度 | 社区数 | 社区大小(min/avg/max) |")
    lines.append("|--------|--------|------|--------|--------|----------------------|")

    for dataset_name in homo_datasets:
        try:
            info = analyze_homo_dataset(dataset_name, data_root)

            # 社区信息
            if info['community_stats']:
                cs = info['community_stats']
                num_comm = cs['num_communities']
                size_str = f"{cs['min_size']}/{cs['avg_size']:.1f}/{cs['max_size']}"
            else:
                num_comm = "-"
                size_str = "-"

            lines.append(f"| {info['name']} | {info['num_nodes']:,} | {info['num_edges']:,} | "
                        f"{info['avg_degree']:.2f} | {num_comm} | {size_str} |")
        except Exception as e:
            lines.append(f"| {dataset_name} | [错误] {str(e)} | - | - | - | - |")

    markdown = "\n".join(lines)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
        print(f"\n[完成] Markdown 表格已保存到: {output_file}")

    return markdown


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='数据集信息统计工具')
    parser.add_argument('--hete_datasets', nargs='+', default=['ACM', 'DBLP', 'IMDB'],
                        help='要分析的异构图数据集列表')
    parser.add_argument('--homo_datasets', nargs='+',
                        default=['amazon', 'com-amazon', 'com-dblp', 'com-youtube', 'com-twitter', 'com-lj'],
                        help='要分析的同构图数据集列表')
    parser.add_argument('--data_root', type=str, default='./datasets',
                        help='数据集根目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 Markdown 文件路径')
    parser.add_argument('--format', choices=['text', 'markdown'], default='text',
                        help='输出格式')
    parser.add_argument('--type', choices=['hete', 'homo', 'all'], default='all',
                        help='输出类型：异构图/同构图/全部')

    args = parser.parse_args()

    if args.format == 'markdown':
        markdown = generate_markdown_table(args.hete_datasets, args.homo_datasets,
                                          args.data_root, args.output)
        if not args.output:
            print("\n" + markdown)
    else:
        if args.type in ['hete', 'all']:
            print_hete_summary(args.hete_datasets, args.data_root)
        if args.type in ['homo', 'all']:
            print_homo_summary(args.homo_datasets, args.data_root)


if __name__ == '__main__':
    main()
