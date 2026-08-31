"""
简化版：生成单个扰动数据集

对每个数据集只生成一个扰动目录，格式与原始 datasets/ 完全一致，
可以直接被 get_cs_dataset 加载。
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import shutil
import json


# ======================================================================
# 节点数量配置
# ======================================================================
DATASET_CONFIG = {
    'ACM': {
        'original_dir': 'datasets/acm',
        'feat_file': 'p_feat.npz',         # 只有 paper 特征参与训练
        'meta_paths': ['pap.npz', 'psp.npz'],
        'edge_files': ['pa.txt', 'ps.txt'],
        'num_nodes': 4019,                  # 只有 paper 节点有特征和标签
        'num_papers': 4019,
        'num_authors': 7167,
        'num_subjects': 60,
    },
    'IMDB_NEW': {
        'original_dir': 'datasets/imdb_new',
        'feat_file': 'm_feat.npz',
        'meta_paths': ['mam.npz', 'mdm.npz'],
        'edge_files': ['ma.txt', 'md.txt'],
        'num_nodes': 4500,
        'num_movies': 4500,
        'num_actors': 6827,
        'num_directors': 2004,
    },
}


# ======================================================================
# 元路径构建（与原始数据集完全一致的方式）
# ======================================================================
def build_metapath(edges, num_src, num_mid):
    """
    从二部图边列表构建 src-mid-src 元路径邻接矩阵。
    与 build_imdb_new.py 中的 build_metapath 完全一致。

    Args:
        edges: shape (E, 2), 每行是 (src_id, mid_id)
        num_src: 源节点数量
        num_mid: 中间节点数量

    Returns:
        [num_src × num_src] 稀疏矩阵
    """
    mid2srcs = defaultdict(list)
    for s, m in edges:
        mid2srcs[m].append(s)

    rows, cols, data = [], [], []
    for m, srcs in mid2srcs.items():
        for i, s1 in enumerate(srcs):
            for s2 in srcs:
                if s1 != s2:
                    rows.append(s1)
                    cols.append(s2)
                    data.append(1)

    mat = sp.coo_matrix((data, (rows, cols)),
                        shape=(num_src, num_src)).tocsr()
    return mat


def build_adj_from_edges(dataset_name, perturbed_edges, node_counts):
    """
    根据扰动后的边构建全局邻接矩阵（包含自环）。

    Args:
        dataset_name: 数据集名称
        perturbed_edges: 扰动后的边列表，如 [pa_edges, ps_edges]
        node_counts: 各类型节点数量，如 [num_papers, num_authors, num_subjects]

    Returns:
        adj: 稀疏邻接矩阵
        edge2type: 边到类型的映射字典
    """
    total_nodes = sum(node_counts)
    rows, cols, etypes = [], [], []

    if dataset_name == 'ACM':
        num_papers, num_authors, num_subjects = node_counts
        pa_edges, ps_edges = perturbed_edges

        # 边类型定义（与原始一致）
        rel2type = {
            'pa': 0, 'ap': 2,
            'ps': 1, 'sp': 3,
            'pp': 4, 'aa': 5, 'ss': 6,
        }

        # paper-author 边
        for p, a in pa_edges:
            gp, ga = p, num_papers + a
            rows += [gp, ga]
            cols += [ga, gp]
            etypes += [rel2type['pa'], rel2type['ap']]

        # paper-subject 边
        for p, s in ps_edges:
            gp, gs = p, num_papers + num_authors + s
            rows += [gp, gs]
            cols += [gs, gp]
            etypes += [rel2type['ps'], rel2type['sp']]

    elif dataset_name == 'IMDB_NEW':
        num_movies, num_actors, num_directors = node_counts
        ma_edges, md_edges = perturbed_edges

        # 边类型定义（与原始一致）
        rel2type = {
            'ma': 0, 'am': 2,
            'md': 1, 'dm': 3,
            'mm': 4, 'aa': 5, 'dd': 6,
        }

        # movie-actor 边
        for m, a in ma_edges:
            gm, ga = m, num_movies + a
            rows += [gm, ga]
            cols += [ga, gm]
            etypes += [rel2type['ma'], rel2type['am']]

        # movie-director 边
        for m, d in md_edges:
            gm, gd = m, num_movies + num_actors + d
            rows += [gm, gd]
            cols += [gd, gm]
            etypes += [rel2type['md'], rel2type['dm']]

    # 添加与节点类型对应的自环：movie/paper=4，actor/author=5，subject/director=6
    self_loop_types = {0: 4, 1: 5, 2: 6}
    for node_type, count in enumerate(node_counts):
        start = sum(node_counts[:node_type])
        for i in range(start, start + count):
            rows.append(i)
            cols.append(i)
            etypes.append(self_loop_types[node_type])

    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)),
                        shape=(total_nodes, total_nodes)).tocsr()
    edge2type = {(r, c): t for r, c, t in zip(rows, cols, etypes)}

    return adj, edge2type


# ======================================================================
# 异构图边扰动
# ======================================================================
def perturb_edges(original_dir, edge_files, num_src_list, num_mid_list,
                  edge_noise_ratio, edge_drop_ratio, seed):
    """
    对异构图的各种边进行扰动（丢弃 + 注入）。

    Args:
        original_dir: 原始数据集目录
        edge_files: 边文件名列表，如 ['pa.txt', 'ps.txt']
        num_src_list: 每种边的源节点数量，如 [4019, 4019]
        num_mid_list: 每种边的中间节点数量，如 [7167, 60]
        edge_noise_ratio: 注入比例
        edge_drop_ratio: 丢弃比例
        seed: 随机种子

    Returns:
        perturbed_edges_list: 每种边的扰动后边列表
    """
    np.random.seed(seed)

    perturbed = []
    for edge_file, num_src, num_mid in zip(edge_files, num_src_list, num_mid_list):
        edges = np.loadtxt(os.path.join(original_dir, edge_file), dtype=int)
        print(f"  {edge_file}: original {len(edges)} edges")

        # 边丢弃
        num_drop = int(len(edges) * edge_drop_ratio)
        if num_drop > 0:
            drop_idx = np.random.choice(len(edges), num_drop, replace=False)
            edges = np.delete(edges, drop_idx, axis=0)
            print(f"    after dropout: {len(edges)} edges")

        # 边注入
        num_inject = int(len(edges) * edge_noise_ratio / max(1 - edge_drop_ratio, 1e-8))
        if num_inject > 0:
            inject = np.column_stack([
                np.random.randint(0, num_src, num_inject),
                np.random.randint(0, num_mid, num_inject)
            ])
            edges = np.vstack([edges, inject])

        # 去重
        edges = np.unique(edges, axis=0)
        print(f"    after injection: {len(edges)} edges")

        perturbed.append(edges)

    return perturbed


# ======================================================================
# 特征扰动
# ======================================================================
def perturb_features(original_dir, feat_file, num_nodes, noise_std, seed):
    """
    添加小幅高斯噪声，同时保持原始特征的数值约束。
    """
    rng = np.random.default_rng(seed + 100)
    feat = sp.load_npz(os.path.join(original_dir, feat_file)).astype(np.float32)

    if feat_file == 'm_feat.npz':
        # IMDB 文本嵌入是带符号的 L2 归一化向量；按维度缩放噪声，
        # 使整行扰动强度约为 noise_std，并恢复 L2 归一化。
        dense = feat.toarray()
        noise = rng.normal(
            0,
            noise_std / np.sqrt(dense.shape[1]),
            dense.shape,
        ).astype(np.float32)
        dense += noise
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        dense /= np.maximum(norms, 1e-12)
        return sp.csr_matrix(dense)

    # ACM 词袋特征只扰动原有非零项，保持稀疏结构和行和约束。
    feat = feat.tocsr(copy=True)
    multipliers = 1.0 + rng.normal(0, noise_std, feat.nnz)
    multipliers = np.maximum(multipliers, 1e-6).astype(np.float32)
    feat.data *= multipliers
    row_sum = np.asarray(feat.sum(axis=1)).ravel()
    inv_row_sum = np.zeros_like(row_sum, dtype=np.float32)
    nonempty = row_sum > 0
    inv_row_sum[nonempty] = 1.0 / row_sum[nonempty]
    feat.data *= np.repeat(inv_row_sum, np.diff(feat.indptr))
    return feat


# ======================================================================
# 派生特征
# ======================================================================
def build_acm_author_features(pa_edges, paper_features, num_authors):
    """根据扰动后的 paper-author 边和 paper 特征重新聚合 author 特征。"""
    relation = sp.coo_matrix(
        (np.ones(len(pa_edges), dtype=np.float32),
         (pa_edges[:, 0], pa_edges[:, 1])),
        shape=(paper_features.shape[0], num_authors),
    ).tocsr()
    return (relation.T @ paper_features).tocsr()


# ======================================================================
# 主函数
# ======================================================================
def generate_perturbed_dataset(dataset_name, perturbation_ratio=0.08, seed=42):
    """
    为数据集生成扰动版本，输出目录格式与原始 datasets/ 完全一致。

    输出文件：
        - meta-path .npz 文件（从扰动后的边重新计算）
        - 特征 .npz 文件（添加高斯噪声）
        - labels.npy（从原始数据集复制）
        - pert_info.json（扰动信息）
    """
    cfg = DATASET_CONFIG[dataset_name]
    original_dir = cfg['original_dir']
    dataset_dir = os.path.join('data', f'{dataset_name}_perturbed')

    print(f"\n{'='*60}")
    print(f"Generating perturbed dataset: {dataset_name}")
    print(f"  Source: {original_dir}")
    print(f"  Output: {dataset_dir}")
    print(f"{'='*60}")

    # 扰动参数
    edge_noise_ratio = perturbation_ratio * 0.4   # 3.2%
    edge_drop_ratio = perturbation_ratio * 0.3    # 2.4%
    feature_noise_std = perturbation_ratio * 0.3   # 0.024

    print(f"\nPerturbation parameters:")
    print(f"  Edge injection: {edge_noise_ratio:.4f} ({edge_noise_ratio*100:.2f}%)")
    print(f"  Edge dropout:   {edge_drop_ratio:.4f} ({edge_drop_ratio*100:.2f}%)")
    print(f"  Feature noise:  {feature_noise_std:.4f}")

    # 创建输出目录，并保留原始目录中的全部文件
    os.makedirs(dataset_dir, exist_ok=True)
    for root, _, files in os.walk(original_dir):
        relative_root = os.path.relpath(root, original_dir)
        target_root = dataset_dir if relative_root == '.' else os.path.join(dataset_dir, relative_root)
        os.makedirs(target_root, exist_ok=True)
        for filename in files:
            shutil.copy2(os.path.join(root, filename), os.path.join(target_root, filename))

    # --- Step 1: 扰动边 ---
    print(f"\n[1/4] Perturbing heterogeneous edges...")

    if dataset_name == 'ACM':
        num_src_list = [cfg['num_papers'], cfg['num_papers']]
        num_mid_list = [cfg['num_authors'], cfg['num_subjects']]
    else:  # IMDB_NEW
        num_src_list = [cfg['num_movies'], cfg['num_movies']]
        num_mid_list = [cfg['num_actors'], cfg['num_directors']]

    perturbed_edges = perturb_edges(
        original_dir, cfg['edge_files'], num_src_list, num_mid_list,
        edge_noise_ratio, edge_drop_ratio, seed
    )

    # 保存扰动后的边
    for edge_file, edges in zip(cfg['edge_files'], perturbed_edges):
        out_path = os.path.join(dataset_dir, edge_file)
        np.savetxt(out_path, edges, fmt='%d')
        print(f"  Saved: {edge_file} ({len(edges)} edges)")

    # --- Step 2: 重新计算元路径 ---
    print(f"\n[2/4] Recomputing meta-paths from perturbed edges...")

    num_src = cfg['num_nodes']
    for meta_path_file, edges, num_mid in zip(
        cfg['meta_paths'], perturbed_edges, num_mid_list
    ):
        mp_adj = build_metapath(edges, num_src, num_mid)
        out_path = os.path.join(dataset_dir, meta_path_file)
        sp.save_npz(out_path, mp_adj)
        print(f"  Saved: {meta_path_file} (nnz={mp_adj.nnz})")

    # --- Step 3: 扰动特征 ---
    print(f"\n[3/4] Perturbing features...")
    perturbed_feat = perturb_features(
        original_dir, cfg['feat_file'], cfg['num_nodes'],
        feature_noise_std, seed
    )
    out_path = os.path.join(dataset_dir, cfg['feat_file'])
    sp.save_npz(out_path, perturbed_feat)
    print(f"  Saved: {cfg['feat_file']} (shape={perturbed_feat.shape})")

    # ACM 的 a_feat 是由 paper 特征按 paper-author 边求和得到的派生特征，
    # 必须使用扰动后的边和 paper 特征重新计算。
    if dataset_name == 'ACM':
        author_feat = build_acm_author_features(
            perturbed_edges[0], perturbed_feat, cfg['num_authors']
        )
        sp.save_npz(os.path.join(dataset_dir, 'a_feat.npz'), author_feat)
        print(f"  Saved: a_feat.npz (shape={author_feat.shape})")

    # --- Step 4: 构建邻接矩阵和边类型映射，复制标签 ---
    print(f"\n[4/4] Building adj and edge2type from perturbed edges...")

    # 构建全局邻接矩阵和边类型映射
    if dataset_name == 'ACM':
        node_counts = [cfg['num_papers'], cfg['num_authors'], cfg['num_subjects']]
    else:  # IMDB_NEW
        node_counts = [cfg['num_movies'], cfg['num_actors'], cfg['num_directors']]

    adj, edge2type = build_adj_from_edges(dataset_name, perturbed_edges, node_counts)

    # 保存邻接矩阵
    adj_path = os.path.join(dataset_dir, 'adj.npz')
    sp.save_npz(adj_path, adj)
    print(f"  Saved: adj.npz (nnz={adj.nnz})")

    # 保存边类型映射
    import pickle
    e2t_path = os.path.join(dataset_dir, 'edge2type.pickle')
    with open(e2t_path, 'wb') as f:
        pickle.dump(edge2type, f)
    print(f"  Saved: edge2type.pickle ({len(edge2type)} edges)")

    # 复制标签和节点类型；ACM 的 a_feat 已在上一步重新生成
    files_to_copy = ['labels.npy', 'node_types.npy']

    for filename in files_to_copy:
        src_path = os.path.join(original_dir, filename)
        dst_path = os.path.join(dataset_dir, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  Copied: {filename}")

    # 保存 pert_info.json
    pert_info = {
        'dataset': dataset_name,
        'source': original_dir,
        'type': 'heterogeneous_edge_perturbation',
        'edge_noise_ratio': edge_noise_ratio,
        'edge_drop_ratio': edge_drop_ratio,
        'feature_noise_std': feature_noise_std,
        'seed': seed,
        'edge_counts': {
            f: len(e) for f, e in zip(cfg['edge_files'], perturbed_edges)
        },
        'metapath_counts': {},
    }
    for mp_file in cfg['meta_paths']:
        mp_path = os.path.join(dataset_dir, mp_file)
        mp_adj = sp.load_npz(mp_path)
        pert_info['metapath_counts'][mp_file] = int(mp_adj.nnz)

    with open(os.path.join(dataset_dir, 'pert_info.json'), 'w') as f:
        json.dump(pert_info, f, indent=2, ensure_ascii=False)
    print(f"  Saved: pert_info.json")

    # --- 汇总 ---
    total_size = sum(
        os.path.getsize(os.path.join(dataset_dir, f))
        for f in os.listdir(dataset_dir)
    )
    print(f"\n{'='*60}")
    print(f"Done! {dataset_name}_perturbed/")
    print(f"  Total size: {total_size / (1024*1024):.2f} MB")
    print(f"  Files ({len(os.listdir(dataset_dir))}):")
    for f in sorted(os.listdir(dataset_dir)):
        fpath = os.path.join(dataset_dir, f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            print(f"    {f} ({size / (1024*1024):.2f} MB)")
        else:
            print(f"    {f} ({size / 1024:.1f} KB)")

    # 对比原始数据集大小
    orig_size = sum(
        os.path.getsize(os.path.join(original_dir, f))
        for f in os.listdir(original_dir)
        if os.path.isfile(os.path.join(original_dir, f))
    )
    print(f"\n  Original size: {orig_size / (1024*1024):.2f} MB")
    print(f"  Ratio: {total_size / orig_size:.2f}x")
    print(f"{'='*60}")


def main():
    print("="*60)
    print("Perturbation Generator")
    print("Generates perturbed datasets in the same format as original")
    print("="*60)

    datasets = ['ACM', 'IMDB_NEW']
    perturbation_ratio = 0.08

    for dataset_name in datasets:
        generate_perturbed_dataset(dataset_name, perturbation_ratio)


if __name__ == '__main__':
    main()
