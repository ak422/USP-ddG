import os
import copy
import argparse
import pandas as pd
import numpy as np
# from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.utils.tensorboard
from torch.utils.data import DataLoader, Dataset
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Selection
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from tqdm.auto import tqdm
import random
import pickle
import lmdb
import re
from colorama import Fore
from collections import defaultdict
from easydict import EasyDict
from src.utils.transforms._base import _get_CB_positions

from src.utils.misc import load_config
from src.utils.data_skempi_mpnn import PaddingCollate
from src.utils.train_mpnn import *

from src.utils.transforms import Compose, SelectAtom,AddAtomNoise, SelectedRegionFixedSizePatch
from src.utils.protein.parsers import parse_biopython_structure
from src.models.model import MoE_ddG_NET
from src.utils.skempi_mpnn import eval_skempi_three_modes, eval_HER2_modes, eval_permutation_modes, eval_multimutation_modes

class CaseDataset(Dataset):
    MAP_SIZE = 500 * (1024 * 1024 * 1024)  # 500GB
    def __init__(self, config):
        super().__init__()
        self.pdb_wt_dir = config.data.pdb_wt_dir
        self.pdb_mt_dir = config.data.pdb_mt_dir
        self.cache_dir = config.data.cache_dir
        self.patch_size = config.data.patch_size

        self.data = []
        self.db_conn = None
        self.entries_cache = os.path.join(self.cache_dir, 'entries.pkl')
        self._load_entries(reset=False)

        self.transform = Compose([
            SelectAtom(config.data.transform[0].resolution),
            AddAtomNoise(config.data.transform[1].noise_backbone,
                         config.data.transform[1].noise_sidechain),
            SelectedRegionFixedSizePatch(config.data.transform[2].select_attr,
                                         config.data.patch_size)
        ])
    def save_mutations(self, mutations):
        data = pd.DataFrame(data=mutations, index=None)
        path = os.path.join(os.path.dirname(self.cache_dir), "7FAE_RBD_Fv_mutations.csv")
        data.to_csv(path)

    def clone_data(self):
        return copy.deepcopy(self.data)

    def _load_entries(self, reset):
        with open(self.entries_cache, 'rb') as f:
            self.entries = pickle.load(f)

    def _load_data(self, pdb_wt_path, pdb_mt_path):
        self.data.append(self._load_structure(pdb_wt_path))
        self.data.append(self._load_structure(pdb_mt_path))

    def _load_structure(self, pdb_path):
        if pdb_path.endswith('.pdb'):
            parser = PDBParser(QUIET=True)
        elif pdb_path.endswith('.cif'):
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError('Unknown file type.')

        model = parser.get_structure(None, pdb_path)[0]
        chains = Selection.unfold_entities(model, 'C')
        random.shuffle(chains)  # shuffle chains，增加数据多样性

        data, seq_map = parse_biopython_structure(model, chains)
        return data, seq_map

    def _parse_mutations(self, mutations):
        parsed = []
        for m in mutations:
            wt, ch, mt = m[0], m[1], m[-1]
            seq = int(m[2:-1])

            if mt == '*':
                for mt_idx in range(20):
                    mt = index_to_one(mt_idx)
                    if mt == wt: continue
                    parsed.append({
                        'chain': ch,
                        'seq': seq,
                        'wt': wt,
                        'mt': mt,
                    })
            else:
                parsed.append({
                    'chain': ch,
                    'seq': seq,
                    'wt': wt,
                    'mt': mt,
                })
        return parsed

    @property
    def lmdb_path(self):
        return os.path.join(self.cache_dir, 'structures.lmdb')

    @property
    def keys_path(self):
        return os.path.join(self.cache_dir, 'keys.pkl')
    def _connect_db(self):
        assert self.db_conn is None
        self.db_conn = lmdb.open(
            self.lmdb_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with open(self.keys_path, 'rb') as f:
            self.db_keys = pickle.load(f)
    def _close_db(self):
        self.db_conn.close()
        self.db_conn = None
        self.db_keys = None
    def _get_from_db(self, pdbcode):
        if self.db_conn is None:
            self._connect_db()
        data = pickle.loads(self.db_conn.begin().get(pdbcode.encode()))   # Made a copy
        return data
    def _compute_degree_centrality(
            self,
            data,
            atom_ty="CB",
            dist_thresh=10,
    ):
        pos_beta_all = _get_CB_positions(data['pos_heavyatom'], data['mask_heavyatom'])
        pairwise_dists = torch.cdist(pos_beta_all, pos_beta_all)
        return torch.sum(pairwise_dists < dist_thresh, dim=-1) - 1
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        pdbcode = entry['#Pdb']

        data_wt, chains_wt = self._get_from_db("wt_" + pdbcode)  # Made a copy
        data_mt, chains_mt = self._get_from_db("mt_" + pdbcode)  # Made a copy

        data_dict_wt = defaultdict(list)
        data_dict_mt = defaultdict(list)
        chains_list = list(data_wt.keys())
        partner = entry['group_ligand'] + entry['group_receptor']
        for i, chain in enumerate(chains_wt):
            if chain not in partner:
                data_wt[i]['is_binding'] = torch.zeros_like(data_wt[i]['aa'])
                data_mt[i]['is_binding'] = torch.zeros_like(data_mt[i]['aa'])
            else:
                data_wt[i]['is_binding'] = torch.ones_like(data_wt[i]['aa'])
                data_mt[i]['is_binding'] = torch.ones_like(data_mt[i]['aa'])

        random.shuffle(chains_list)
        for i in chains_list:
            if isinstance(data_wt[i], EasyDict):
                for k, v in data_wt[i].items():
                    data_dict_wt[k].append(v)
            if isinstance(data_mt[i], EasyDict):
                for k, v in data_mt[i].items():
                    data_dict_mt[k].append(v)

        for k, v in data_dict_wt.items():
            data_dict_wt[k] = torch.cat(data_dict_wt[k], dim=0)
        for k, v in data_dict_mt.items():
            data_dict_mt[k] = torch.cat(data_dict_mt[k], dim=0)

        # centrality
        # pos_heavyatom: ['N', 'CA', 'C', 'O', 'CB']
        data_dict_wt['centrality'] = self._compute_degree_centrality(data_dict_wt)  # CB原子
        data_dict_mt['centrality'] = self._compute_degree_centrality(data_dict_mt)  # CB原子

        keys = {'id', 'complex_PPI', 'mutstr', 'num_muts', '#Pdb', 'ddG','cath_domain', 'wt_scores_cycle', 'mut_scores_cycle'}
        for k in keys:
            data_dict_wt[k] = data_dict_mt[k] = entry[k]
        data_dict_wt['inter_energy'] = entry['inter_energy_wt']
        data_dict_mt['inter_energy'] = entry['inter_energy_mt']

        assert len(entry['mutations']) == torch.sum(data_dict_wt['aa'] != data_dict_mt['aa']), f"ID={data_dict_wt['id']},{len(entry['mutations'])},{torch.sum(data_dict_wt['aa'] != data_dict_mt['aa'])}"
        data_dict_wt['mut_flag'] = data_dict_mt['mut_flag'] = (data_dict_wt['aa'] != data_dict_mt['aa'])

        if self.transform is not None:
            data_dict_wt, _ = self.transform(data_dict_wt)
            data_dict_mt, _ = self.transform(data_dict_mt)

        return {"wt": data_dict_wt,
                "mt": data_dict_mt,
                }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--output_results', type=str, default='case_results.csv')
    parser.add_argument('--output_metrics', type=str, default='case_metrics.csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_cvfolds', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    ckpt = []
    config, config_name = load_config(args.config)
    for checkpoint in config.checkpoints:
        ckpt.append(torch.load(checkpoint, map_location=args.device))
    config_model = ckpt[0]['config']
    print(config_model)

    # load model
    cv_mgr = CrossValidation(model_factory=MoE_ddG_NET,
                             config=config_model,
                             early_stoppingdir=config_model.early_stoppingdir,
                             num_cvfolds=config_model.train.num_cvfolds).to(args.device)
    # Data
    dataset = CaseDataset(
        config=config
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=PaddingCollate(config.data.patch_size),
        num_workers=args.num_workers,
    )

    results = []
    for fold in range(config_model.train.num_cvfolds):
        results_fold = []
        for i in range(len(ckpt)):
            cv_mgr.load_state_dict(ckpt[i]['model'], )
            model, _, _, _ = cv_mgr.get(fold)
            model.eval()
            with torch.no_grad():
                for batch in tqdm(loader, desc=f'\033[0;37;42m Fold {fold + 1}/{config_model.train.num_cvfolds} Model {i + 1}/{len(ckpt)} \033[0m', dynamic_ncols=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.RESET)):
                    # Prepare data
                    batch = recursive_to(batch, args.device)
                    # Forward pass
                    output_dict = model.inference(batch)
                    for pdbcode, complex_PPI, mutstr, ddg_true, ddg_pred in zip(batch["wt"]['#Pdb'], batch["wt"]['complex_PPI'], batch["wt"]['mutstr'], output_dict['ddG_true'], output_dict['ddG_pred']):
                        results_fold.append({
                            'pdbcode': pdbcode,
                            'complex_PPI': complex_PPI,
                            'mutstr': mutstr,
                            'num_muts': len(mutstr.split(',')),
                            'ddG': ddg_true.item(),
                            'ddG_pred': ddg_pred.item(),
                            'fold': fold,
                        })
        results_fold = pd.DataFrame(results_fold)
        results.extend(results_fold.to_dict(orient='records'))
    results = pd.DataFrame(results)
    results['datasets'] = 'case_study'
    results.replace("1.00E+96", "1E96", inplace=True)
    results.replace("1.00E+50", "1E50", inplace=True)
    results.to_csv(args.output_results, index=False)

    results = pd.read_csv(args.output_results)
    # 显示所有列,保留4位小数
    pd.set_option('display.max_columns', None)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.width', 250)  # 设置打印宽度(**重要**)
    pd.options.display.float_format = '{:.4f}'.format
    if 'S285' in config.data.cache_dir:
        # S285
        print('Case study S285.')
        results = results.groupby(['pdbcode','mutstr','datasets']).agg(ddG_pred_mean=("ddG_pred", "mean"),
                                                ddG=("ddG", "mean"),
                                                num_muts=("num_muts", "mean")).reset_index()
        results['ddG_pred'] = -results['ddG_pred_mean']
        print('# Length of Results: ', results.shape[0])
        df_metrics = eval_skempi_three_modes(results)
        df_metrics.to_csv(args.output_metrics)
        df_metrics_string = df_metrics.to_string()
        print('Results:\n', df_metrics_string)
    elif 'CR6261' in config.data.cache_dir:
        # CR6261
        print('Case study CR6261.')
        results = results.groupby(['pdbcode','mutstr','datasets']).agg(ddG_pred_mean=("ddG_pred", "mean"),
                                                ddG=("ddG", "mean"),
                                                num_muts=("num_muts", "mean")).reset_index()
        results['ddG_pred'] = -results['ddG_pred_mean']
        print('# Length of Results: ', results.shape[0])
        df_metrics = eval_skempi_three_modes(results)
        df_metrics.to_csv(args.output_metrics)
        df_metrics_string = df_metrics.to_string()
        print('Results:\n', df_metrics_string)

        df_metrics = eval_multimutation_modes(results)
        print('#Length of multi-point: ', df_metrics['Count'].iloc[:-1].sum())
        df_metrics_string = df_metrics.to_string()
        print(df_metrics_string)
    elif 'HER2' in config.data.cache_dir:
        # HER2
        print('Case study HER2.')
        for fold in range(config_model.train.num_cvfolds):
            results_fold = results[results['fold'] == fold].copy()
            results_fold = results_fold.groupby(['pdbcode', 'mutstr', 'datasets']).agg(ddG_pred_mean=("ddG_pred", "mean"),
                                                                                        ddG=("ddG", "mean"),
                                                                                        num_muts=("num_muts", "mean")).reset_index()
            results_fold['ddG_pred'] = results_fold['ddG_pred_mean']
            df_metrics = eval_permutation_modes(results_fold)
            print(f'Results of [Fold-{fold}]:\n', df_metrics)

        results = results.groupby(['pdbcode','mutstr', 'datasets']).agg(ddG_pred_mean=("ddG_pred", "mean"),
                                                ddG=("ddG", "mean"),
                                                num_muts=("num_muts", "mean")).reset_index()
        results['ddG_pred'] = results['ddG_pred_mean']
        df_metrics = eval_HER2_modes(results)
        df_metrics.to_csv(args.output_metrics)
        print('# Length of Results: ', results.shape[0])
        print('Results:\n', df_metrics)
    if 'Demo' in config.data.cache_dir:
        # Demo
        print('Case study Demo.')
        results = results.groupby(['pdbcode','mutstr','datasets']).agg(ddG_pred_mean=("ddG_pred", "mean"),
                                                ddG=("ddG", "mean"),
                                                num_muts=("num_muts", "mean")).reset_index()
        results['ddG_pred'] = -results['ddG_pred_mean']
        print('# Length of Results: ', results.shape[0])
        df_metrics = eval_skempi_three_modes(results)
        df_metrics.to_csv(args.output_metrics)
        df_metrics_string = df_metrics.to_string()
        print('Results:\n', df_metrics_string)