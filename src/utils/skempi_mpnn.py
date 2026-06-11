import functools
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm
import torch
import os
import pickle
import random
import logging

from src.utils.misc import inf_iterator, BlackHole
from src.utils.data_skempi_mpnn import PaddingCollate
from src.utils.transforms import get_transform
from src.datasets import SkempiDataset_lmdb

from torch.utils.data.sampler import Sampler
from collections import defaultdict
from collections import Counter

# class ClassSequentialSampler(Sampler):
#     def __init__(self, labels, shuffle_classes=False, shuffle_samples=True, logger=None):
#         """
#         Args:
#             labels (list): 数据集的结构域标签列表，如 [0, 1, 2, 0, 1, 2, ...]
#             shuffle_classes (bool): 是否打乱类别顺序（默认 True）
#             shuffle_samples (bool): 是否打乱当前类别内的样本顺序（默认 True）
#         """
#         self.labels = labels
#         self.shuffle_classes = shuffle_classes
#         self.shuffle_samples = shuffle_samples
#
#         # 构建类别到样本索引的映射
#         self.label_to_indices = defaultdict(list)
#         for idx, label in enumerate(labels):
#             self.label_to_indices[label].append(idx)
#         # {cath-class: lable-index}
#         cath_index_dict = {3: 0, 2: 1, 1: 2, 4: 3, 0: 4, 6: 5}
#         # self.cath_index = [[0], [2], [1], [3], [4], [5]]
#         self.cath_index = [[5], [1], [4], [0], [2], [3]]
#         import time
#         # random.seed(int(time.time()))
#         # random.shuffle(self.cath_index)
#         logger.info(self.cath_index)
#
#         self.num_classes = len(self.cath_index)
#         self.num_samples = len(labels)
#
#     def __iter__(self):
#         # 1. 打乱类别顺序（如果启用）
#         if self.shuffle_classes:
#             np.random.shuffle(self.cath_index)
#
#         # 2. 遍历每个类别，并采样当前类别的样本
#         indices = []
#         for cls in self.cath_index:
#             cls_indices = []
#             for cl in cls:
#                 cls_indices.extend(self.label_to_indices[cl])
#             if self.shuffle_samples:
#                 np.random.shuffle(cls_indices)
#             indices.extend(cls_indices)
#
#         return iter(indices)
#
#     def __len__(self):
#         return self.num_samples

class ClassSequentialSampler(Sampler):
    def __init__(self, labels, shuffle_classes=False, shuffle_samples=True, logger=None):
        self.labels = labels
        self.shuffle_classes = shuffle_classes
        self.shuffle_samples = shuffle_samples

        # 构建类别到样本索引的映射
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        cath_index_dict = {3: 0, 2: 1, 1: 2, 4: 3, 0: 4, 6: 5}
        self.cath_index = [[1], [5], [2], [0], [3], [4]]
        # import time
        # random.seed(int(time.time()))
        # random.shuffle(self.cath_index)
        # logger.info(self.cath_index)
        reverse_dict = {v: k for k, v in cath_index_dict.items()}
        mapped_keys = [[reverse_dict[idx[0]]] for idx in self.cath_index]
        logger.info(mapped_keys)

        self.num_classes = len(self.cath_index)
        self.num_samples = len(labels)
        self.logger = logger

        # ===== 新增：epoch 计数器 =====
        self.epoch_count = 0

    def __iter__(self):
        # epoch 计数递增
        self.epoch_count += 1
        is_first_epoch = (self.epoch_count == 1)

        if self.shuffle_classes:
            np.random.shuffle(self.cath_index)

        # 遍历每个类别，并采样当前类别的样本
        indices = []
        epoch_order_log = []  # 用于收集第一个 epoch 的详细顺序

        for cls in self.cath_index:
            cls_indices = []
            class_labels = []
            for cl in cls:
                cls_indices.extend(self.label_to_indices[cl])
                class_labels.append(cl)

            if self.shuffle_samples and not is_first_epoch:
                np.random.shuffle(cls_indices)
            # if self.shuffle_samples:
            #     np.random.shuffle(cls_indices)

            # 记录第一个 epoch 的类别和样本信息
            if is_first_epoch:
                epoch_order_log.append({
                    'class': cls,
                    'class_labels': class_labels,
                    'num_samples': len(cls_indices),
                    'sample_indices': cls_indices.copy()
                })

            indices.extend(cls_indices)

        # ===== 新增：第一个 epoch 将完整顺序写入文件 =====
        if is_first_epoch:
            # 构建输出内容
            lines = []
            lines.append("=" * 70)
            lines.append(f"[ClassSequentialSampler] 第 1 个 Epoch 的采样顺序")
            lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"类别遍历顺序 (cath_index): {self.cath_index}")
            lines.append(f"总样本数: {len(indices)}")
            lines.append("-" * 70)

            all_sample_indices = []
            for i, info in enumerate(epoch_order_log):
                lines.append(f"Step {i + 1}: CATH 类别 {info['class']} (标签 {info['class_labels']}), "
                             f"样本数 {info['num_samples']}")
                lines.append(f"  样本索引: {info['sample_indices']}")
                all_sample_indices.extend(info['sample_indices'])

            lines.append("-" * 70)
            lines.append(f"完整采样索引序列 (共 {len(all_sample_indices)} 个):")
            # 每行输出 20 个索引，便于阅读
            for i in range(0, len(all_sample_indices), 20):
                chunk = all_sample_indices[i:i + 20]
                lines.append(f"  [{i:5d}:{min(i + 20, len(all_sample_indices)):5d}] {chunk}")

            lines.append("=" * 70)

            output_text = "\n".join(lines)

            # ===== 核心修改：提取 logger 日志目录 =====
            log_dir = None
            if self.logger is not None:
                for handler in self.logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        log_dir = os.path.dirname(handler.baseFilename)
                        break

            # 回退保护：若 logger 无 FileHandler，使用当前工作目录
            save_dir = log_dir if log_dir is not None else os.getcwd()

            # 文件名固定即可（因目录已按运行隔离），或加时间戳更保险
            output_filename = "epoch1_sample_order.txt"
            output_path = os.path.join(save_dir, output_filename)


            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output_text)
                print(f"\n[已保存] 第一个 epoch 的样本顺序已写入: {output_path}")
            except Exception as e:
                print(f"\n[保存失败] 无法写入文件 {output_path}: {e}")

            # 同时写入 logger
            if self.logger is not None:
                self.logger.info(f"[Epoch 1] Sample order saved to {output_path}, "
                                 f"total {len(all_sample_indices)} samples")

        return iter(indices)

    def __len__(self):
        return self.num_samples


class SkempiDatasetManager(object):
    def __init__(self, config, split_seed, num_cvfolds, device, num_workers=4, logger=BlackHole()):
        super().__init__()
        self.config = config
        self.num_cvfolds = num_cvfolds
        self.train_loaders = []
        self.val_loaders = []
        self.chains = []
        self.logger = logger
        self.device = device
        self.num_workers = num_workers
        self.split_seed = split_seed
        for fold in range(num_cvfolds):
            train_loader, val_loader = self.init_loaders(fold)
            self.train_loaders.append(train_loader)
            self.val_loaders.append(val_loader)

    def init_loaders(self, fold):
        config = self.config
        dataset_ = functools.partial(
            SkempiDataset_lmdb,
            csv_path = config.data.csv_path,
            pdb_wt_dir = config.data.pdb_wt_dir,
            pdb_mt_dir=config.data.pdb_mt_dir,
            prior_dir=config.data.prior_dir,
            cache_dir = config.data.cache_dir,
            device = self.device,
            num_cvfolds = self.num_cvfolds,
            cvfold_index = fold,
            split_seed = self.split_seed,
            is_single=config.data.is_single,
            cath_fold=config.data.cath_fold,
            PPIformer=config.data.PPIformer,
        )

        train_dataset = dataset_(split='train',transform = get_transform(config.data.train.transform))
        val_dataset = dataset_(split='val',transform = get_transform(config.data.val.transform))
        
        train_cplx = set([e['complex_PPI'] for e in train_dataset.entries])
        val_cplx = set([e['complex_PPI'] for e in val_dataset.entries])
        leakage = train_cplx.intersection(val_cplx)
        assert len(leakage) == 0, f'data leakage {leakage}'
        # cath_label_train = [e['cath_label_index'] for e in train_dataset.entries]

        # import csv
        # tm_score_dict = {}
        # df = pd.read_csv('./data/SKEMPI2/TM-score.csv', sep=',')
        # df.replace("1.00E+96", "1E96", inplace=True)
        # df.replace("1.00E+50", "1E50", inplace=True)
        # for i, row in df.iterrows():
        #     pdb = row['pdb']
        #     tm_score = float(row['TM-score'])
        #     tm_score_dict[pdb] = tm_score

        # 对 entries 排序
        train_dataset.entries.sort(
            key=lambda x: tm_score_dict.get(x.get('complex', ''), 0),
            reverse=True
        )
        sorted_data =  train_dataset.entries
        cath_label_train = [e['cath_label_index'] for e in sorted_data]


        sampler = ClassSequentialSampler(
            labels=cath_label_train,
            shuffle_classes=False,  # 每个 epoch 随机打乱类别顺序
            shuffle_samples=True,  # 每个类别内的样本顺序随机
            logger=self.logger,
        )

        # 按cath_label采样
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            collate_fn=PaddingCollate(config.data.train.transform[2].patch_size),
            # shuffle=True,
            sampler=sampler,
            shuffle=False,    # 采样器已经控制顺序，无需再 shuffle
            num_workers=self.num_workers
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.train.batch_size,
            shuffle=False,
            collate_fn=PaddingCollate(config.data.val.transform[2].patch_size),
            num_workers=self.num_workers
        )

        self.logger.info('Fold %d: Train %d, Val %d, All %d' % (fold + 1, len(train_dataset), len(val_dataset), (len(train_dataset)+len(val_dataset))))

        return train_loader, val_loader

    def get_train_loader(self, fold):
        return self.train_loaders[fold]

    def get_val_loader(self, fold):
        return self.val_loaders[fold]

def overall_correlations(df):
    pearson = df[['ddG', 'ddG_pred']].corr('pearson').iloc[0,1]
    spearman = df[['ddG', 'ddG_pred']].corr('spearman').iloc[0,1]
    return {
        'overall_pearson': pearson, 
        'overall_spearman': spearman,
    }

def perprotein_correlations(df, return_details=False, complex_threshold=8):
    corr_table = []
    for cplx in df['protein_group'].sort_values().unique():
        df_cplx = df.query(f'protein_group == "{cplx}"')
        corr_table.append({
            'protein_group': cplx,
            'count': df_cplx.shape[0],
            'pearson': df_cplx[['ddG', 'ddG_pred']].corr('pearson').iloc[0,1],
            'spearman': df_cplx[['ddG', 'ddG_pred']].corr('spearman').iloc[0,1],
            'auroc': overall_auroc(df_cplx)['auroc'],
        })
    corr_table = pd.DataFrame(corr_table)
    average = corr_table[['pearson', 'spearman',  'auroc']].mean()
    out = [{
        'protein_group': 'average',
        'count': '-',
        'pearson': average['pearson'],
        'spearman': average['spearman'],
        'auroc': average['auroc'],
    }]
    out = pd.DataFrame(out)
    pd_out = pd.concat([corr_table, out], ignore_index = True)
    if return_details:
        return pd_out, corr_table
    else:
        return pd_out

def percomplex_correlations(df, return_details=False):
    corr_table = []
    for cplx in np.sort(df['complex_PPI'].unique()):
    # for cplx in np.sort(df['protein_group'].unique()):
        df_cplx = df.query(f'complex_PPI == "{cplx}"')
        # df_cplx = df.query(f'protein_group == "{cplx}"')
        if len(df_cplx) < 10:
            continue
        corr_table.append({
            'complex_PPI': cplx,
            # 'protein_group': cplx,
            'pearson': df_cplx[['ddG', 'ddG_pred']].corr('pearson').iloc[0,1],
            'spearman': df_cplx[['ddG', 'ddG_pred']].corr('spearman').iloc[0,1],
        })
    corr_table = pd.DataFrame(corr_table)
    average = corr_table[['pearson', 'spearman']].mean()
    out = {
        'perPPI_pearson': average['pearson'],
        'perPPI_spearman': average['spearman'],
        # 'perprotein_pearson': average['pearson'],
        # 'perprotein_spearman': average['spearman'],
    }
    if return_details:
        return out, corr_table
    else:
        return out

def permutation_correlations(df, return_details=False):
    corr_all = []
    corr_all.append({
        'N_mut': 'All',
        'Count': df.shape[0],
        'PearsonR': df[['ddG', 'ddG_pred']].corr('pearson').iloc[0,1],
        'SpearmanR': df[['ddG', 'ddG_pred']].corr('spearman').iloc[0,1],
    })
    corr_all = pd.DataFrame(corr_all)

    corr_num = []
    for cplx in np.sort(df['num_muts'].unique()):
        df_cplx = df.query(f'num_muts == {cplx}')
        if len(df_cplx) == 1:
            corr_num.append({
                'N_mut': int(cplx),
                'Count': df_cplx.shape[0],
                'PearsonR': np.nan,
                'SpearmanR': np.nan,
            })
            continue
        corr_num.append({
            'N_mut': int(cplx),
            'Count': df_cplx.shape[0],
            'PearsonR': df_cplx[['ddG', 'ddG_pred']].corr('pearson').iloc[0,1],
            'SpearmanR': df_cplx[['ddG', 'ddG_pred']].corr('spearman').iloc[0,1],
        })
    corr_num = pd.DataFrame(corr_num)

    pd_out = pd.concat([corr_num, corr_all], ignore_index = True)
    return pd_out

def overall_auroc(df):
    try:
        # Only one class present
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            score = roc_auc_score(
                (df['ddG'] > 0).to_numpy(),
                df['ddG_pred'].to_numpy()
            )
    except ValueError:
        score = np.nan
    return {
        'auroc': score,
    }

def overall_rmse_mae(df):
    true = df['ddG'].to_numpy()
    pred = df['ddG_pred'].to_numpy()[:, None]
    reg = LinearRegression().fit(pred, true)
    pred_corrected = reg.predict(pred)
    rmse = np.sqrt( ((true - pred_corrected) ** 2).mean() )
    mae = np.abs(true - pred_corrected).mean()
    pred_neg = df['ddG_pred'] < 0
    real_neg = df['ddG'] < 0
    precision = precision_score(real_neg, pred_neg, zero_division=0),
    recall = recall_score(real_neg, pred_neg, zero_division=0),
    return {
        'rmse': rmse,
        'mae': mae,
    }

def analyze_all_results(df):
    datasets = df['datasets'].unique()
    funcs = {
        'SKEMPI2': [
                    overall_correlations,
                    overall_rmse_mae,
                    overall_auroc,
                    percomplex_correlations,
                    ],
        'case_study': [
                    overall_correlations,
                    overall_rmse_mae,
                    overall_auroc,
                    ]
    }
    analysis = []
    for dataset in datasets:
        assert dataset in ['SKEMPI2', 'case_study']
        df_this = df[df['datasets'] == dataset]
        result = {
            'dataset': dataset,
        }
        for f in funcs[dataset]:
            result.update(f(df_this))
        analysis.append(result)
    analysis = pd.DataFrame(analysis)
    return analysis

def eval_skempi(df_items, mode, ddg_cutoff=None):
    assert mode in ('all', 'single', 'multiple')
    if mode == 'single':
        df_items = df_items.query('num_muts == 1')
    elif mode == 'multiple':
        df_items = df_items.query('num_muts > 1')

    if ddg_cutoff is not None:
        df_items = df_items.query(f"ddG >= {-ddg_cutoff} and ddG <= {ddg_cutoff}")

    df_metrics = analyze_all_results(df_items)
    df_metrics['mode'] = mode
    return df_metrics

def eval_skempi_three_modes(results, ddg_cutoff=None):
    df_all = eval_skempi(results, mode='all', ddg_cutoff=ddg_cutoff)
    df_single = eval_skempi(results, mode='single', ddg_cutoff=ddg_cutoff)
    df_multiple = eval_skempi(results, mode='multiple', ddg_cutoff=ddg_cutoff)
    df_metrics = pd.concat([df_all, df_single, df_multiple], axis=0)
    df_metrics.reset_index(inplace=True, drop=True)
    return df_metrics

def eval_HER2_modes(results, ddg_cutoff=None):
    df_all = eval_skempi(results, mode='all', ddg_cutoff=ddg_cutoff)
    df_metrics = pd.concat([df_all], axis=0)
    df_metrics.reset_index(inplace=True, drop=True)
    return df_metrics

def eval_perprotein_modes(df_items, ddg_cutoff=None):
    if ddg_cutoff is not None:
        df_items = df_items.query(f"ddG >= {-ddg_cutoff} and ddG <= {ddg_cutoff}")
    # evalueate per_protein
    perprotein_metrics = perprotein_correlations(df_items)
    return perprotein_metrics

def eval_permutation_modes(df_items, ddg_cutoff=None):
    if ddg_cutoff is not None:
        df_items = df_items.query(f"ddG >= {-ddg_cutoff} and ddG <= {ddg_cutoff}")
    permutation_metrics = permutation_correlations(df_items)
    return permutation_metrics

def eval_multimutation_modes(df_items, ddg_cutoff=None):
    if ddg_cutoff is not None:
        df_items = df_items.query(f"ddG >= {-ddg_cutoff} and ddG <= {ddg_cutoff}")
    # filter out single point mutations
    df_items = df_items.query(f'num_muts > 1')
    permutation_metrics = permutation_correlations(df_items)
    return permutation_metrics
