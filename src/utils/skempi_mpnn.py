import functools
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm
import os
import logging
from pathlib import Path
import subprocess
import re
import multiprocessing as mp

from src.utils.misc import inf_iterator, BlackHole
from src.utils.data_skempi_mpnn import PaddingCollate
from src.utils.transforms import get_transform
from src.datasets import SkempiDataset_lmdb

from torch.utils.data.sampler import Sampler
from collections import defaultdict

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
        self.epoch = 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        is_first_epoch = (self.epoch == 1)

        # if self.shuffle_classes:
        #     np.random.shuffle(self.cath_index)
        if self.shuffle_classes and not is_first_epoch:  # not shuffle in epoch 1
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

            if self.shuffle_samples and not is_first_epoch:  # not shuffle in epoch 1
                np.random.shuffle(cls_indices)
            # if self.shuffle_samples:
            #     np.random.shuffle(cls_indices)

            indices.extend(cls_indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples


def _get_pdb_path(complex_name, pdb_dir):
    pdb_path = os.path.join(pdb_dir, f"{complex_name}.pdb")
    if os.path.exists(pdb_path):
        return pdb_path
    pdb_path = os.path.join(pdb_dir, f"{complex_name.upper()}.pdb")
    if os.path.exists(pdb_path):
        return pdb_path
    return None


def _run_tmalign(pdb1, pdb2):
    try:
        result = subprocess.run(
            ['TMalign', pdb1, pdb2],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout
        # Parse the first TM-score (normalized by length of Chain_1, i.e., query)
        match = re.search(r'TM-score=\s+([0-9.]+)', output)
        if match:
            return float(match.group(1))
        return 0.0
    except Exception:
        return 0.0


def _compute_max_tmscore(args):
    tc, val_pdb_paths, PDB_dir = args
    tp = _get_pdb_path(tc, PDB_dir)
    if tp is None:
        return (tc, 0.0, False)
    max_tm = 0.0
    for vp in val_pdb_paths:
        tm = _run_tmalign(tp, vp)
        if tm > max_tm:
            max_tm = tm
    return (tc, max_tm, True)


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

        train_complex = set([e['complex'] for e in train_dataset.entries])
        val_complex = set([e['complex'] for e in val_dataset.entries])
        PDB_dir = Path(self.config.data.pdb_wt_dir).parent.joinpath('PDBs')

        # Build val PDB path list
        val_pdb_paths = []
        for vc in val_complex:
            vp = _get_pdb_path(vc, PDB_dir)
            if vp is not None:
                val_pdb_paths.append(vp)
            else:
                self.logger.warning(f'Val PDB not found for complex: {vc}')

        # Prepare arguments for multiprocessing
        train_list = list(train_complex)
        args_list = [(tc, val_pdb_paths, PDB_dir) for tc in train_list]
        num_workers = min(mp.cpu_count(), len(args_list)) if len(args_list) > 0 else 1

        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_compute_max_tmscore, args_list),
                total=len(args_list),
                desc=f'Computing TM-scores for Fold-{fold+1}'
            ))

        tm_score_dict = {}
        for tc, max_tm, found in results:
            if not found:
                self.logger.warning(f'Train PDB not found for complex: {tc}')
            tm_score_dict[tc] = max_tm

        # 对 entries 排序
        train_dataset.entries.sort(
            key=lambda x: tm_score_dict.get(x.get('complex', ''), 0),
            reverse=True
        )
        sorted_data =  train_dataset.entries
        cath_label_train = [e['cath_label_index'] for e in sorted_data]

        sampler = ClassSequentialSampler(
            labels=cath_label_train,
            shuffle_classes=True,  # 每个 epoch 随机打乱类别顺序
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
            num_workers=self.num_workers,
            drop_last=True
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
