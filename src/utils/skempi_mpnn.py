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

from src.utils.misc import inf_iterator, BlackHole
from src.utils.data_skempi_mpnn import PaddingCollate
from src.utils.transforms import get_transform
from src.datasets import SkempiDataset_lmdb

from torch.utils.data.sampler import Sampler
from collections import defaultdict
from collections import Counter

class ClassSequentialSampler(Sampler):
    def __init__(self, labels, shuffle_classes=False, shuffle_samples=True, logger=None):
        """
        Args:
            labels (list): 数据集的结构域标签列表，如 [0, 1, 2, 0, 1, 2, ...]
            shuffle_classes (bool): 是否打乱类别顺序（默认 True）
            shuffle_samples (bool): 是否打乱当前类别内的样本顺序（默认 True）
        """
        self.labels = labels
        self.shuffle_classes = shuffle_classes
        self.shuffle_samples = shuffle_samples

        # 构建类别到样本索引的映射
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        # {cath-class: lable-index}
        cath_index_dict = {3: 0, 2: 1, 1: 2, 4: 3, 0: 4, 6: 5}
        self.cath_index = [[0], [2], [1], [3], [4], [5]]

        self.num_classes = len(self.cath_index)
        self.num_samples = len(labels)

    def __iter__(self):
        # 1. 打乱类别顺序（如果启用）
        if self.shuffle_classes:
            np.random.shuffle(self.cath_index)

        # 2. 遍历每个类别，并采样当前类别的样本
        indices = []
        for cls in self.cath_index:
            cls_indices = []
            for cl in cls:
                cls_indices.extend(self.label_to_indices[cl])
            if self.shuffle_samples:
                np.random.shuffle(cls_indices)
            indices.extend(cls_indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples


class SkempiDatasetManager(object):
    def __init__(self, config, split_seed, num_cvfolds, num_workers=4, logger=BlackHole()):
        super().__init__()
        self.config = config
        self.num_cvfolds = num_cvfolds
        self.train_loaders = []
        self.val_loaders = []
        self.chains = []
        self.logger = logger
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
        cath_label_train = [e['cath_label_index'] for e in train_dataset.entries]

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
            collate_fn=PaddingCollate(),
            # shuffle=True,
            sampler=sampler,
            shuffle=False,    # 采样器已经控制顺序，无需再 shuffle
            num_workers=self.num_workers
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.train.batch_size,
            shuffle=False,
            collate_fn=PaddingCollate(),
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
