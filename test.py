import os
import copy
import argparse
from colorama import Fore
from sklearn import preprocessing
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.tensorboard
from torch.utils.data import DataLoader, Dataset
from Bio import BiopythonDeprecationWarning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", BiopythonDeprecationWarning)
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB import Selection
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.Polypeptide import index_to_one, one_to_index
from tqdm.auto import tqdm
import random
import pickle
import lmdb
from collections import defaultdict
from easydict import EasyDict
from src.utils.transforms._base import _get_CB_positions

from src.utils.misc import load_config, seed_all
from src.utils.data_skempi_mpnn import PaddingCollate
from src.utils.train_mpnn import *

from src.utils.transforms import Compose, SelectAtom,AddAtomNoise, SelectedRegionFixedSizePatch
from src.utils.protein.parsers import parse_biopython_structure
from src.models.model import MoE_ddG_NET
from src.utils.skempi_mpnn import  SkempiDatasetManager, eval_skempi_three_modes, eval_perprotein_modes, eval_multimutation_modes

def process_ddg(results):
    mut_dict = defaultdict(list)
    for i, row in results.iterrows():
        pdb_id = row['pdb_id']
        mutstr = row['mutstr'].split(",")
        mutstr.sort()
        mutstr = '_'.join(mutstr)
        complex_str = pdb_id + '_' + mutstr
        mut_dict[complex_str].append(row['ddG_pred'])

    for key, values in mut_dict.items():
        mut_dict[key] = np.mean(values)

    ddg_list = []
    for i, row in results.iterrows():
        pdb_id = row['pdb_id']
        mutstr = row['mutstr'].split(",")
        mutstr.sort()
        mutstr = '_'.join(mutstr)
        complex_str = pdb_id + '_' + mutstr
        ddg_list.append(mut_dict[complex_str])

    results['ddG_pred'] = pd.DataFrame(ddg_list)
    results = pd.DataFrame(results)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--output_results', type=str, default='results_USP_ddG.csv')
    parser.add_argument('--output_metrics', type=str, default='skempi2_metrics.csv')
    parser.add_argument('--output_perprotein', type=str, default='perprotein_metrics.csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--early_stoppingdir', type=str, default='./early_stopping')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_cvfolds', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()
    config, _ = load_config(args.config)
    print(config)

    # Model
    ckpt = []
    for checkpoint in config.checkpoints:
        ckpt.append(torch.load(checkpoint, map_location=args.device, weights_only=False))
    config_model = ckpt[0]['config']
    seed_all(config_model.train.seed)

    cv_mgr = CrossValidation(
        model_factory=MoE_ddG_NET,
        config=config_model,
        early_stoppingdir=config_model.early_stoppingdir,
        num_cvfolds=config_model.train.num_cvfolds
    ).to(args.device)

    dataset_mgr = SkempiDatasetManager(
        # config_model,
        config,
        split_seed=config_model.train.seed,
        num_cvfolds=config_model.train.num_cvfolds,
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
                for j, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc=f'\033[0;37;42m Fold {fold+1}/{config_model.train.num_cvfolds} Model {i+1}/{len(ckpt)} \033[0m', dynamic_ncols=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.RESET))):
                    # Prepare data
                    batch = recursive_to(batch, args.device)

                    # Forward pass
                    output_dict = model.inference(batch)
                    for pdbcode, protein_group, complex_PPI, mutstr, ddg_true, ddg_pred in zip(batch["wt"]['#Pdb'],batch["wt"]['protein_group'],batch["wt"]['complex_PPI'],batch["wt"]['mutstr'],output_dict['ddG_true'],output_dict['ddG_pred']):
                        results_fold.append({
                            'pdbcode': pdbcode,
                            'protein_group': protein_group,
                            'complex_PPI': complex_PPI,
                            'mutstr': mutstr,
                            'num_muts': len(mutstr.split(',')),
                            'ddG': ddg_true.item(),
                            'ddG_pred': ddg_pred.item()
                        })
        results_fold = pd.DataFrame(results_fold)
        results_fold = results_fold.groupby(['pdbcode', 'complex_PPI', 'protein_group']).agg(ddG_pred_mean=("ddG_pred", "mean"),
                                                                              ddG=("ddG", "mean"),
                                                                              num_muts=("num_muts", "mean")).reset_index()
        results_fold['ddG_pred'] = results_fold['ddG_pred_mean']
        results.extend(results_fold.to_dict(orient='records'))

    results = pd.DataFrame(results)
    results.replace("1.00E+96", "1E96", inplace=True)
    results.replace("1.00E+50", "1E50", inplace=True)
    results.to_csv(args.output_results, index=False)

    results = pd.read_csv(args.output_results)
    results = results.groupby(['pdbcode', 'complex_PPI', 'protein_group']).agg(ddG_pred_mean=("ddG_pred", "mean"),
                                                         ddG=("ddG", "mean"),
                                                         num_muts=("num_muts", "mean")).reset_index()
    results['ddG_pred'] = results['ddG_pred_mean']
    # 显示所有列,保留4位小数
    pd.set_option('display.max_columns', None)
    pd.options.display.float_format = '{:.4f}'.format
    results['datasets'] = 'SKEMPI2'

    df_metrics = eval_skempi_three_modes(results)
    df_metrics_string = df_metrics.to_string()
    print('#Length of results:', results.shape[0])
    print('Results:\n%s', df_metrics_string)

    # eval multi-point mutations
    df_metrics = eval_multimutation_modes(results)
    print('#Length of multi-point: ', df_metrics['Count'].iloc[:-1].sum())
    df_metrics_string = df_metrics.to_string()
    print(df_metrics_string)

    if config_model.data.PPIformer == True:
        perprotein_metrics = eval_perprotein_modes(results)
        perprotein_metrics.to_csv(args.output_perprotein, index=False)
        print(perprotein_metrics)

