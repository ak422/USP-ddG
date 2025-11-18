import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import sys
import subprocess
import argparse
sys.path.append('../..')
import os
import copy
import random
import pandas as pd  # 先调用pandas, 再调用torch，否则出现error: version `GLIBCXX_3.4.29' not found
import pickle
import math
import torch
import lmdb
import json
from colorama import Fore
import numpy as np
from pymol import cmd
import h5py
import re
import shutil
from pathlib import Path
import networkx as nx
from functools import partial
import multiprocessing as mp
from itertools import dropwhile
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Selection
from Bio.PDB.Polypeptide import one_to_index
from src.utils.protein.dihedral_chi import get_dihedral
from src.utils.misc import  current_milli_time
from easydict import EasyDict
from collections import defaultdict
from typing import Mapping, List, Dict, Tuple, Optional
from joblib import Parallel, delayed, cpu_count
from scipy import stats, special
from src.utils.transforms._base import _get_CB_positions
from src.datasets.InterfaceResidues import interfaceResidues
from src.utils.protein.constants import AA

from src.utils.protein.parsers import parse_biopython_structure
from transformers import AutoTokenizer, AutoModelForMaskedLM

def pdb2dict(pdb_path):
    ca_pattern = re.compile(
        "^ATOM\s{2,6}\d{1,5}\s{2}CA\s[\sA]([A-Z]{3})\s([\s\w])|^HETATM\s{0,4}\d{1,5}\s{2}CA\s[\sA](MSE)\s([\s\w])")
    chain_dict = dict()
    chain_list = []
    with open(pdb_path, 'r') as fp:
        for line in fp.read().splitlines():
            if line.startswith("ENDMDL"):
                break
            match_list = ca_pattern.findall(line)
            if match_list:
                resn = match_list[0][0] + match_list[0][2]
                if not AA.is_aa(resn):
                    continue

                restype = AA(resn)
                if restype == AA.UNK:
                    continue

                chain = match_list[0][1] + match_list[0][3]
                if chain in chain_dict:
                    chain_dict[chain] += AA.three2one(resn)
                else:
                    chain_dict[chain] = AA.three2one(resn)
                    chain_list.append(chain)
    return chain_dict

def cath_domain_stat(pdb_chain, cath_domains_dict, cath_level, zero_domain=pd.DataFrame()):
    cath_domains = []
    for key in cath_domains_dict.keys():
        if pdb_chain in key:
            cath_domain = '.'.join(cath_domains_dict[key][0:cath_level])
            cath_domains.append(cath_domain) # read 4 classes of CATH
    if len(cath_domains) == 0:
            if ~zero_domain.empty and pdb_chain in list(zero_domain.index):
                cath_domain = zero_domain.loc[pdb_chain]['cath_classification'].split('.')[0:cath_level]
                cath_domain = '.'.join(cath_domain)
                cath_domains.append(cath_domain) # No domains in CATH
            else:
                cath_domain = '.'.join(['0'] * cath_level)
                cath_domains.append(cath_domain) # No domains in CATH

    return list(set(cath_domains))

def skempi2_chainset(df, block_list):
    def _parse_mut(mut_name):
        wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
        mutseq = int(mut_name[2:-1])
        return {
            'wt': wt_type,
            'mt': mt_type,
            'chain': mutchain,
            'resseq': mutseq,
            'icode': ' ',
            'name': mut_name
        }
    def _parse_reverse_mut(mut_name):
        wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
        mutseq = int(mut_name[2:-1])
        mut_name = "".join([mt_type, mutchain, mut_name[2:-1], wt_type])
        return {
            'wt': mt_type,
            'mt': wt_type,
            'chain': mutchain,
            'resseq': mutseq,
            'icode': ' ',
            'name': mut_name
        }

    skempi2_domain = defaultdict(list)
    pdb_chain_set = set()

    for i, row in df.iterrows():
        pdbcode = row['#Pdb']
        complex = row['complex']
        group1 = row['Partner1']
        group2 = row['Partner2']

        if complex.upper() in block_list:
            continue
        if not np.isfinite(row['ddG']):
            continue

        mut_str = row['Mutation(s)_cleaned']
        mut_list = set(mut_str.split(','))

        if row['Label'] == "forward":
            muts = list(map(_parse_mut, mut_list))
            mut_sets = set([mut['chain'] for mut in muts])
        else:
            # 处理M1707的逆突变
            muts = list(map(_parse_reverse_mut, mut_list))
            mut_str = ",".join([mut['name'] for mut in muts])
            row['ddG'] = -row['ddG']
            mut_sets = set([mut['chain'] for mut in muts])

        chain_group = list(row['Partner1'] + row['Partner2'])

        # cath processing
        for chain in chain_group:
            pdb_chain = (complex.lower() + chain.upper())
            pdb_chain_set.add(pdb_chain)
            skempi2_domain[pdbcode].append({pdb_chain: ""})

    return pdb_chain_set

def _get_structure(pdb_path, interface, chains, pdbcode, esm2, flag):
    parser = PDBParser(QUIET=True)
    model = parser.get_structure(None, pdb_path)[0]

    data, chains_map = parse_biopython_structure(model, interface, esm2, chains)
    _structures = {}
    _structures[flag + "_" +pdbcode] = (data, chains_map)

    return _structures

def write_domain_graph(prior_dir, cath_domain_complex):
    # 根据cath_domain_complex 构造domains连接图
    cath_domain_reverse = defaultdict(list)
    for complex_id, domains in cath_domain_complex.items():
        [cath_domain_reverse[domain].append(complex_id) for domain in list(set(domains))]

    cath_domain_node = {}
    for i, (key, value) in enumerate(cath_domain_reverse.items()):
        cath_domain_node[key] = i
    # 输出节点列表
    node_path = os.path.join(prior_dir, 'node.csv')
    with open(node_path, 'w') as f:
        f.write('cath_domains\n')
        [f.write('{0}\n'.format(key)) for key, value in cath_domain_node.items()]
    # 构造邻接矩阵
    cath_domain_edge = pd.DataFrame(columns=['Source', 'Target'])
    adjacency_matrix = np.zeros((len(cath_domain_node), len(cath_domain_node)))
    for key, values in cath_domain_complex.items():
        if len(values) > 1:
            for i in values:
                for j in values:
                    if i > j:
                        cath_domain_edge = pd.concat([cath_domain_edge, pd.DataFrame([{'Source': i, 'Target': j}])],
                                                     ignore_index=True)
    edge_path = os.path.join(prior_dir, 'edge.csv')
    cath_domain_edge.to_csv(edge_path)

def split_cath_domains(prior_dir, df, csv_path):
    node_path = os.path.join(prior_dir, 'node.csv')
    edge_path = os.path.join(prior_dir, 'edge.csv')

    nodes_df = pd.read_csv(node_path)
    nodes_df.sort_values(by="cath_domains", inplace=True, ascending=False)

    edges_df = pd.read_csv(edge_path).iloc[:, 1:].drop_duplicates()

    G = nx.Graph()

    for index, row in nodes_df.iterrows():
        G.add_node(row['cath_domains'])

    for index, row in edges_df.iterrows():
        G.add_edge(row['Source'], row['Target'])

    # 求取连通分量
    connected_components = list(nx.connected_components(G))
    # 降序排序
    sorted_components = sorted(connected_components, key=len, reverse=True)

    # 打印各个连通分量的节点数及节点名称
    for i, component in enumerate(sorted_components):
        print(f"[Connected component {i + 1}]: #nodes = {len(component)}, nodes = {component}")

    total_nodes = G.number_of_nodes()
    print(f"#All domains: {total_nodes}")

    train_nodes = set()
    test_nodes = set()

    # 将连通分量的节点依降序放入训练集中，直到其近似占70%
    target_train_size = int(total_nodes * 0.7)
    current_train_size = 0

    for component in sorted_components:
        # 检查加上当前连通分量后，训练集是否会超出目标大小
        if current_train_size + len(component) <= target_train_size:
            train_nodes.update(component)  # 将整个连通分量加入训练集
            current_train_size += len(component)
        else:
            test_nodes.update(component)  # 将剩余连通分量加入测试集

    # 输出结果为csv格式
    train_df = pd.DataFrame(train_nodes, columns=['Node'])
    test_df = pd.DataFrame(test_nodes, columns=['Node'])
    print(f"#Domains of train：{train_df.shape[0]}, #Domains of test：{test_df.shape[0]}")

    cath_json_path = os.path.join(prior_dir, 'cath_domain_complex.json')
    with open(cath_json_path, 'r') as f:
        cath_domain_complex = json.load(f)

    # load cath_domain_complex_1
    cath_json_path = os.path.join(prior_dir, 'cath_domain_complex_1.json')
    with open(cath_json_path, 'r') as f:
        cath_domain_complex_1 = json.load(f)

    train_path = os.path.join(prior_dir, 'train_nodes.csv')
    test_path = os.path.join(prior_dir, 'test_nodes.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    if 'cath_fold' in df.columns:
        df.drop('cath_fold', axis=1, inplace=True)

    new_df = pd.DataFrame(columns=df.columns)
    cath_label_dict = {'3':0, '2':1, '1':2, '4':3, '0':4, '6':5}   # 结构域映射关系表

    # 多个cath class条件下, 默认取第一个class结构域
    cath_class = {}
    for complex, domain in cath_domain_complex_1.items():
        if complex not in cath_class.keys():
            cath_class[complex] = domain[0]    # 默认取第一个
            continue

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f'\033[0;37;42m CATH label generating...\033[0m'):
        complex = row['complex']
        if complex.upper() in {'1KBH'}:
            continue
        if not np.isfinite(row['ddG']):
            continue

        if complex.lower() not in cath_domain_complex.keys():
            continue
        cath_domain_list = cath_domain_complex[complex.lower()]

        for cath_domain in cath_domain_list:
            if cath_domain in list(train_df['Node']):
                new_df.loc[len(new_df)] = row
                new_df.loc[len(new_df) - 1, 'cath_fold'] = 'train'
                new_df.loc[len(new_df) - 1, 'cath_label_index'] = cath_label_dict[cath_class[complex.lower()]]
            elif cath_domain in list(test_df['Node']):
                new_df.loc[len(new_df)] = row
                new_df.loc[len(new_df) - 1, 'cath_fold'] = 'val'
                new_df.loc[len(new_df) - 1, 'cath_label_index'] = cath_label_dict[cath_class[complex.lower()]]
            break
    new_df.to_csv(csv_path, index=False)

    # read cath_label from file
    source_df_path = csv_path.parent.parent / csv_path.name
    source_df = pd.read_csv(source_df_path, sep=',')
    if csv_path.exists():
        df = pd.read_csv(csv_path, sep=',')
        df.replace("1.00E+96", "1E96", inplace=True)
        df.replace("1.00E+50", "1E50", inplace=True)

        # 将source 复制到 csv_path
        mapping_dict = {}
        for _, row in source_df.iterrows():
            pdb_id = row['#Pdb']
            mapping_dict[pdb_id] = {col: row[col] for col in ['cath_label_index']}
        # 根据映射字典填充数据
        for idx, row in df.iterrows():
            pdb_id = row['#Pdb']
            if pdb_id in mapping_dict:
                for col in ['cath_label_index']:
                    df.at[idx, col] = mapping_dict[pdb_id][col]
        df.to_csv(csv_path, index=False)
    else:
        shutil.copy(source_df_path, csv_path.parent)
        df = pd.read_csv(csv_path, sep=',')
        df.replace("1.00E+96", "1E96", inplace=True)
        df.replace("1.00E+50", "1E50", inplace=True)

    print('train size:', len(new_df[new_df['cath_fold'].str.contains('train', na=False, case=False)]))
    print('test size:', len(new_df[new_df['cath_fold'].str.contains('val', na=False, case=False)]))
def load_category_entries(csv_path, cath_domain_path, prior_dir, pdb_wt_dir, pdb_mt_dir, block_list={'1KBH'}):
    source_df_path = csv_path.parent.parent / csv_path.name
    source_df = pd.read_csv(source_df_path, sep=',')
    if csv_path.exists():
        df = pd.read_csv(csv_path, sep=',')
        df.replace("1.00E+96", "1E96", inplace=True)
        df.replace("1.00E+50", "1E50", inplace=True)

        # 将source 复制到 csv_path
        mapping_dict = {}
        for _, row in source_df.iterrows():
            pdb_id = row['#Pdb']
            mapping_dict[pdb_id] = {col: row[col] for col in ['wt_scores_cycle', 'mut_scores_cycle']}
        # 根据映射字典填充数据
        for idx, row in df.iterrows():
            pdb_id = row['#Pdb']
            if pdb_id in mapping_dict:
                for col in ['wt_scores_cycle', 'mut_scores_cycle']:
                    df.at[idx, col] = mapping_dict[pdb_id][col]
    else:
        shutil.copy(source_df_path, csv_path.parent)
        df = pd.read_csv(csv_path, sep=',')
        df.replace("1.00E+96", "1E96", inplace=True)
        df.replace("1.00E+50", "1E50", inplace=True)
    # ----------------------- #
    # generate cath_domain_compex
    cath_domain_complex_4 = defaultdict(list)
    cath_domain_complex_1 = defaultdict(list)
    cath_domain_protein_4 = defaultdict(list)
    cath_domain_protein_1 = defaultdict(list)

    # 1. 读取skempi2 构造 chain set
    pdb_chain_set = skempi2_chainset(df, block_list)

    # 构造cath字典
    cath_file = Path(prior_dir, 'cath-domain-list.txt')
    if not cath_file.exists():
        cath_file.parent.mkdir(exist_ok=True, parents=True)
        subprocess.run(f'wget ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt -P {cath_file.parent}', shell=True)

    cath_domains_dict = defaultdict(list)
    with open(cath_file, "r") as file:
        for line in dropwhile(lambda line: line.startswith('#'), file):
            cath_domains_dict[line.split()[0]] = line.split()[1:5]
    # 输出蛋白质结构域列表
    json_str = json.dumps(cath_domains_dict, indent=4)
    cath_json_path = os.path.join(prior_dir, 'cath_domains_dict.json')
    with open(cath_json_path, 'w') as json_file:
        json_file.write(json_str)

    # 输出0结构域链
    cath_domain_zero = []
    cath_domain_all = []  # 所有结构域
    zero_domain_path = os.path.join(prior_dir, 'zero_domain_chain.csv')
    all_domain_path = os.path.join(prior_dir, 'all_domain_chain.csv')
    cath_level = 4
    for pdb_chain in tqdm(pdb_chain_set, desc=f'\033[0;37;42m Chains2domains\033[0m'):
        cath_domain_cal = cath_domain_stat(pdb_chain, cath_domains_dict, cath_level)
        if cath_domain_cal == ['0.0.0.0']:
            cath_domain_zero.append([f'{pdb_chain}', '0.0.0.0'])
        cath_domain_all.append([f'{pdb_chain}', cath_domain_cal])
    data = pd.DataFrame(data=cath_domain_zero, columns=['pdb_chain', 'cath_classification'])
    data2 = pd.DataFrame(data=cath_domain_all, columns=['pdb_chain', 'cath_classification'])
    data.to_csv(zero_domain_path, index=None)
    data2.to_csv(all_domain_path, index=None)

    # 构造complex 的cath domain字典
    zero_domain = pd.read_csv(zero_domain_path, sep=',', index_col=0)
    zero_domain_foldseek_path = os.path.join(prior_dir.parent, 'zero_domain_foldseek.csv')
    zero_domain_foldseek = pd.read_csv(zero_domain_foldseek_path, sep=',', index_col=0)
    for pdb_chain in tqdm(pdb_chain_set, desc=f'\033[0;37;42m Chains2domains\033[0m'):
        complex_id = pdb_chain[:4]
        # Cath domains:class (C), architecture (A), topology (T) and homologous superfamily (H)
        cath_level = 4
        cath_domain_cal = cath_domain_stat(pdb_chain, cath_domains_dict, cath_level, zero_domain)
        cath_domain_complex_4[complex_id].extend(cath_domain_cal)
        cath_domain_protein_4[pdb_chain].extend(cath_domain_cal)

        cath_level = 1
        cath_domain_cal = cath_domain_stat(pdb_chain, cath_domains_dict, cath_level, zero_domain_foldseek)
        # cath_domain_cal = cath_domain_stat(pdb_chain, cath_domains_dict, cath_level, zero_domain)
        cath_domain_complex_1[complex_id].extend(cath_domain_cal)
        cath_domain_protein_1[pdb_chain].extend(cath_domain_cal)

    # 输出complex-4 domains
    for complex, cath_domain in cath_domain_complex_4.items():
        cath_domain_complex_4[complex] = list(set(cath_domain))
    json_str = json.dumps(cath_domain_complex_4, indent=4)
    with open(cath_domain_path, 'w') as json_file:
        json_file.write(json_str)

    # 输出protein-4 domains
    for protein, cath_domain in cath_domain_protein_4.items():
        cath_domain_protein_4[protein] = list(set(cath_domain))
    json_str = json.dumps(cath_domain_protein_4, indent=4)
    cath_domain_path = Path(prior_dir, 'cath_domain_protein_4.json')
    with open(cath_domain_path, 'w') as json_file:
        json_file.write(json_str)

    # 输出complex-1 domains
    for complex, cath_domain in cath_domain_complex_1.items():
        cath_domain_complex_1[complex] = list(set(cath_domain))
    json_str = json.dumps(cath_domain_complex_1, indent=4)
    cath_domain_path = Path(prior_dir, 'cath_domain_complex_1.json')
    with open(cath_domain_path, 'w') as json_file:
        json_file.write(json_str)

    # 输出protein-1 domains
    for protein, cath_domain in cath_domain_protein_1.items():
        cath_domain_protein_1[protein] = list(set(cath_domain))
    json_str = json.dumps(cath_domain_protein_1, indent=4)
    cath_domain_path = Path(prior_dir, 'cath_domain_protein_1.json')
    with open(cath_domain_path, 'w') as json_file:
        json_file.write(json_str)

    # 根据cath_domain_complex 构造domains连接图, output: node.csv, edge.csv,
    write_domain_graph(prior_dir, cath_domain_complex_4)
    # ---------------- #

    # data splitting
    split_cath_domains(prior_dir, df, csv_path)

    # load cath_domain_complex
    cath_json_path = os.path.join(prior_dir, 'cath_domain_complex_1.json')
    with open(cath_json_path, 'r') as f:
        cath_domain_complex_1 = json.load(f)

    cath_domain_dict = {}
    cath_classes = ['0', '1', '2', '3', '4', '6'] # 0: unannotated
    mlb  = MultiLabelBinarizer(classes = cath_classes)
    MultiLabels = mlb.fit_transform(cath_domain_complex_1.values())
    for i, (key,value) in enumerate(cath_domain_complex_1.items()):
        cath_domain_dict[key] = MultiLabels[i]

    def _parse_mut(mut_name):
        wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
        mutseq = int(mut_name[2:-1])
        return {
            'wt': wt_type,
            'mt': mt_type,
            'chain': mutchain,
            'resseq': mutseq,
            'icode': ' ',
            'name': mut_name
        }
    def _parse_reverse_mut(mut_name):
        wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
        mutseq = int(mut_name[2:-1])
        mut_name = "".join([mt_type, mutchain, mut_name[2:-1], wt_type])
        return {
            'wt': mt_type,
            'mt': wt_type,
            'chain': mutchain,
            'resseq': mutseq,
            'icode': ' ',
            'name': mut_name
        }

    entries = []
    df = pd.read_csv(csv_path, sep=',')
    df.replace("1.00E+96", "1E96", inplace=True)
    df.replace("1.00E+50", "1E50", inplace=True)

    for (i, row) in tqdm(df.iterrows(), total=len(df), desc=f'\033[0;37;42m Generating entries \033[0m'):
        pdbcode = row['#Pdb']
        complex = row['complex']
        group1 = row['Partner1']
        group2 = row['Partner2']
        complex_PPI = row['complex_PPI']
        if 'S285' not in csv_path.name and 'HER2' not in csv_path.name and 'CR6261' not in csv_path.name and 'Demo' not in csv_path.name:
            protein_1 = row['Protein 1']
            protein_2 = row['Protein 2']
        else:
            protein_1 = 'NAN'
            protein_2 = 'NAN'

        if not np.isfinite(row['ddG']):
            continue

        if complex.upper() in block_list:
            continue

        mut_str = row['Mutation(s)_cleaned']
        mut_list = set(mut_str.split(','))

        if row['Label'] == "forward":
            muts = list(map(_parse_mut, mut_list))
        else:
            # 处理M1707的逆突变
            muts = list(map(_parse_reverse_mut, mut_list))
            mut_str = ",".join([mut['name'] for mut in muts])
            row['ddG'] = -row['ddG']

        if muts[0]['chain'] in group1:
            group_ligand, group_receptor = group1, group2
        else:
            group_ligand, group_receptor = group2, group1

        if row['Label'] == "forward":
            pdb_wt_path = os.path.join(pdb_wt_dir, '{}.pdb'.format(pdbcode.upper()))
            pdb_mt_path = os.path.join(pdb_mt_dir, '{}.pdb'.format(pdbcode.upper()))
        else:    # 处理M1707的逆突变
            pdb_mt_path = os.path.join(pdb_wt_dir, '{}.pdb'.format(pdbcode.upper()))
            pdb_wt_path = os.path.join(pdb_mt_dir, '{}.pdb'.format(pdbcode.upper()))

        if not os.path.exists(pdb_wt_path) or not os.path.exists(pdb_mt_path):
            continue


        # read Interaction Energy of foldX
        foldX_dg_wt = Path(prior_dir, 'wildtype_ddg', f'Interaction_{pdbcode}.txt')
        foldX_dg_mt = Path(prior_dir, 'mutant_ddg', f'Interaction_{pdbcode}.txt')
        with open(foldX_dg_wt, "r") as file:
            lines = file.readlines()
            for line in lines:
                if pdbcode.upper() in line:
                    inter_energy_wt = line.split()[6:15] + line.split()[17:19] + [line.split()[20]] + line.split()[23:26]
                    # inter_energy_wt = line.split()[6:19] + line.split()[20:26]   # Backbone Clash is not considered
                    inter_energy_wt = np.array(list(map(float, inter_energy_wt)))
                    break
        with open(foldX_dg_mt, "r") as file:
            lines = file.readlines()
            for line in lines:
                if pdbcode.upper() in line:
                    inter_energy_mt = line.split()[6:15] + line.split()[17:19] + [line.split()[20]] + line.split()[23:26]
                    # inter_energy_mt = line.split()[6:19] + line.split()[20:26]  # Backbone Clash is not considered
                    inter_energy_mt = np.array(list(map(float, inter_energy_mt)))
                    break

        entry = {
            'id': i,
            '#Pdb': pdbcode.upper(),
            'complex': complex,
            'complex_PPI': complex_PPI,
            'wt_scores_cycle': np.float32(row['wt_scores_cycle']),
            'mut_scores_cycle': np.float32(row['mut_scores_cycle']),
            'mutstr': mut_str,
            'num_muts': len(muts),
            'mut_type': len(muts) > 1,
            'protein_group': protein_1,
            'group_ligand': list(group_ligand),
            'group_receptor': list(group_receptor),
            'mutations': muts,
            'ddG': np.float32(row['ddG']),
            # 'ddG_FoldX': np.float32(row['ddG_FoldX']),         # ddG_FoldX: Interaction
            'cath_domain': cath_domain_dict[complex.lower()],
            'cath_label_index': row['cath_label_index'],
            'pdb_wt_path': pdb_wt_path,
            'pdb_mt_path': pdb_mt_path,
            'inter_energy_wt': inter_energy_wt,
            'inter_energy_mt': inter_energy_mt,
        }
        entries.append(entry)
    return entries

def _process_structure(pdb_wt_path, pdb_mt_path, esm2_path, ligand, receptor, pdbcode) -> Optional[Dict]:
    structures = defaultdict(dict)
    parser = PDBParser(QUIET=True)
    model = parser.get_structure(None, pdb_wt_path)[0]
    chains = Selection.unfold_entities(model, 'C')

    # delete invalid chain
    for i, chain in enumerate(chains):
        if chain.id == " ":
            del chains[i]

    interface_wt = []
    cmd.load(pdb_wt_path)  # 载入目录中的蛋白或配体
    for chain_A in ligand:
        for chain_B in receptor:
            if chain_A == chain_B:
                continue
            rVal, ans = interfaceResidues(pdbcode, 'c. ' + chain_A, 'c. ' + chain_B)
            mapp = {'chA': chain_A, 'chB': chain_B}
            for line in ans:
                linee = line.strip().split('_')
                resid = linee[0]  # 残基序号
                chainn = mapp[linee[1]]  # 链标识符cha, 或者 链标识符chb
                inter = '{}_{}'.format(chainn, resid)  # 如 inter: E_I_E_20  后面2个表示结合位点的残基和残基序号
                if inter not in interface_wt:
                    interface_wt.append(inter)
    cmd.delete('all')

    interface_mt = []
    cmd.load(pdb_mt_path)  # 载入目录中的蛋白或配体
    for chain_A in ligand:
        for chain_B in receptor:
            if chain_A == chain_B:
                continue
            rVal, ans = interfaceResidues(pdbcode, 'c. ' + chain_A, 'c. ' + chain_B)
            mapp = {'chA': chain_A, 'chB': chain_B}
            for line in ans:
                linee = line.strip().split('_')
                resid = linee[0]  # 残基序号
                chainn = mapp[linee[1]]  # 链标识符cha, 或者 链标识符chb
                inter = '{}_{}'.format(chainn, resid)  # 如 inter: E_I_E_20  后面2个表示结合位点的残基和残基序号
                if inter not in interface_mt:
                    interface_mt.append(inter)
    cmd.delete('all')

    # esm2 embeddings
    h5_file = h5py.File(esm2_path, "r")
    esm2_wt = h5_file['wt'][pdbcode]
    esm2_mt = h5_file['mt'][pdbcode]

    structures.update(_get_structure(pdb_wt_path, interface_wt, chains, pdbcode, esm2_wt, "wt"))
    structures.update(_get_structure(pdb_mt_path, interface_mt, chains, pdbcode, esm2_mt, "mt"))
    return structures

def generate_esm2(pdbcode_list, esm2_650M_cache):
    tokenizer = AutoTokenizer.from_pretrained("../../data/esm2_t33_650M_UR50D",  device_map = "auto", max_memory={0: "8GiB"},offload_folder='offload')
    esm_model = AutoModelForMaskedLM.from_pretrained("../../data/esm2_t33_650M_UR50D", device_map="cpu",  output_hidden_states=True)
    # esm_model = AutoModelForMaskedLM.from_pretrained("../../data/esm2_t33_650M_UR50D",  device_map = "auto", max_memory={0: "8GiB"},  output_hidden_states=True)

    with h5py.File(esm2_650M_cache, "w") as h5file:
        h5file.create_group('wt')
        h5file.create_group('mt')
        for (pdbcode, pdb_wt_path, pdb_mt_path, ligand, receptor) in tqdm(pdbcode_list, desc=f'\033[0;37;42m ESM2 embeddings\033[0m'):
            h5file['wt'].create_group(f'{pdbcode}')
            h5file['mt'].create_group(f'{pdbcode}')
            # generate sequences
            sequences = pdb2dict(pdb_wt_path)
            for chain, sequence in sequences.items():
                # sequences_tokenized = tokenizer(sequence, return_tensors="pt")['input_ids'].squeeze(0).to('cuda:0')
                sequences_tokenized = tokenizer(sequence, return_tensors="pt")['input_ids'].squeeze(0)
                with torch.no_grad():
                    # last_hidden_states = esm_model(sequences_tokenized.unsqueeze(0)).last_hidden_state[:, 1:-1, :]
                    last_hidden_states = esm_model(sequences_tokenized.unsqueeze(0)).hidden_states[-1][:, 1:-1, :]
                data_wt_chain = last_hidden_states.squeeze(0)
                h5file['wt'][pdbcode].create_dataset(f"{chain}", data=data_wt_chain.cpu().numpy(), dtype='f')

            sequences = pdb2dict(pdb_mt_path)
            for chain, sequence in sequences.items():
                # sequences_tokenized = tokenizer(sequence, return_tensors="pt")['input_ids'].squeeze(0).to('cuda:0')
                sequences_tokenized = tokenizer(sequence, return_tensors="pt")['input_ids'].squeeze(0)
                with torch.no_grad():
                    # last_hidden_states = esm_model(sequences_tokenized.unsqueeze(0)).last_hidden_state[:, 1:-1, :]
                    last_hidden_states = esm_model(sequences_tokenized.unsqueeze(0)).hidden_states[-1][:, 1:-1, :]
                data_mt_chain = last_hidden_states.squeeze(0)
                h5file['mt'][pdbcode].create_dataset(f"{chain}", data=data_mt_chain.cpu().numpy(), dtype='f')

class SkempiDataset_lmdb(Dataset):
    MAP_SIZE = 500 * (1024 * 1024 * 1024)  # 500GB
    def __init__(
        self, 
        csv_path,
        pdb_wt_dir,
        pdb_mt_dir,
        cache_dir,
        prior_dir,
        cvfold_index=0, 
        num_cvfolds=3, 
        split='train', 
        split_seed=2022,
        num_preprocess_jobs=math.floor(cpu_count() * 0.6),
        transform=None, 
        blocklist=frozenset({'1KBH'}), 
        reset=False,

        is_single=2,  # 0:single,1:multiple,2:overall
        cath_fold=False,
        PPIformer=False,
        GearBind=False,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.prior_dir = Path(prior_dir)
        self.pdb_wt_dir = Path(pdb_wt_dir)
        self.pdb_mt_dir = Path(pdb_mt_dir)
        self.cache_dir = Path(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        self.blocklist = blocklist
        self.transform = transform
        self.cvfold_index = cvfold_index
        self.num_cvfolds = num_cvfolds
        assert split in ('train', 'val', 'all')
        self.split = split
        self.split_seed = split_seed
        self.num_preprocess_jobs = num_preprocess_jobs

        self.entries_cache = Path(cache_dir, 'entries.pkl')
        self.interfaces_cache = Path(prior_dir, 'interface.json')
        self.cath_domain_cache = Path(prior_dir, 'cath_domain_complex.json')
        self.esm2_650M_cache = Path(prior_dir, 'ESM2_650M.hdf5')
        self.entries = None
        self.entries_full = None
        # ak422
        self.is_single = is_single
        self.cath_fold = cath_fold
        self.PPIformer = PPIformer
        self.GearBind = GearBind
        self._load_entries(reset)

        self.structures_cache = os.path.join(cache_dir, 'structures.lmdb')
        self.structures = None
        # Structure cache
        self.db_conn = None
        self.db_keys: Optional[List[PdbCodeType]] = None
        self._load_structures(reset)

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()    # 按数据集划分
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

        # 1. 转成 DataFrame
        df = pd.DataFrame(self.entries_full)
        # 2. 保存为 CSV（index=False 表示不额外保存行索引）
        df.to_csv(Path(self.prior_dir, "entries_full.csv"), index=False, encoding="utf-8-sig")

        complex_to_entries = {}
        for e in self.entries_full:
            if e['complex_PPI'] not in complex_to_entries:
                complex_to_entries[e['complex_PPI']] = []
            complex_to_entries[e['complex_PPI']].append(e)

        complex_list = sorted(complex_to_entries.keys())
        if self.cath_fold == False and self.PPIformer == False: # fixed seed for reproducibility, same as Prompt-DDG and BA-DDG
            random.Random(3745754758).shuffle(complex_list)
        else:
            random.Random(self.split_seed).shuffle(complex_list)

        split_size = math.ceil(len(complex_list) / self.num_cvfolds)
        complex_splits = [
            complex_list[i*split_size : (i+1)*split_size] 
            for i in range(self.num_cvfolds)
        ]

        if self.num_cvfolds == 1:
            train_split = sum(complex_splits, start=[])
            val_split = []
        else:
            val_split = complex_splits.pop(self.cvfold_index)
            train_split = sum(complex_splits, start=[])

        # ak422
        if self.cath_fold == True:
            train_split = []
            val_split = []
            df = pd.read_csv(self.csv_path, sep=',')
            df.replace("1.00E+96", "1E96", inplace=True)
            df.replace("1.00E+50", "1E50", inplace=True)

            for i, row in df.iterrows():
                # if row['complex_PPI'] in complex_list:
                if row['cath_fold'] == 'train':
                    train_split.append(row['complex_PPI'])
                elif row['cath_fold'] == 'val':
                    val_split.append(row['complex_PPI'])
            train_split = list(set(train_split))
            val_split = list(set(val_split))

            # added for 3-fold
            if self.num_cvfolds > 1:
                train_split.sort()
                random.Random(self.split_seed).shuffle(train_split)
                split_size = math.ceil(len(train_split) / self.num_cvfolds)
                complex_splits = [
                    train_split[i * split_size: (i + 1) * split_size]
                    for i in range(self.num_cvfolds)
                ]
                complex_splits.pop(self.cvfold_index)
                train_split = sum(complex_splits, start=[])
        elif self.PPIformer == True:
            train_split = []
            val_split = []
            df = pd.read_csv(self.csv_path, sep=',')
            df.replace("1.00E+96", "1E96", inplace=True)
            df.replace("1.00E+50", "1E50", inplace=True)

            for i, row in df.iterrows():
                # if row['complex_PPI'] in complex_list:
                if row['PPIformer'] == 'train':
                    train_split.append(row['complex_PPI'])
                elif row['PPIformer'] == 'val':
                    val_split.append(row['complex_PPI'])
            train_split = list(set(train_split))
            val_split = list(set(val_split))

            # added for 3-fold
            if self.num_cvfolds > 1:
                train_split.sort()
                random.Random(self.split_seed).shuffle(train_split)
                split_size = math.ceil(len(train_split) / self.num_cvfolds)
                complex_splits = [
                    train_split[i * split_size: (i + 1) * split_size]
                    for i in range(self.num_cvfolds)
                ]
                complex_splits.pop(self.cvfold_index)
                train_split = sum(complex_splits, start=[])
        elif self.GearBind == True:
            train_split = [[],[],[],[],[]]
            val_split = [[],[],[],[],[]]
            df = pd.read_csv(self.csv_path, sep=',')
            df.replace("1.00E+96", "1E96", inplace=True)
            df.replace("1.00E+50", "1E50", inplace=True)

            for i, row in df.iterrows():
                if row['complex_PPI'] in complex_list:
                    val_split[int(row['fold'])].append(row['complex_PPI'])

            val_split = [list(set(val)) for val in val_split]
            for i, val in enumerate(val_split):
                complement = list(set(complex_list) - set(val))
                train_split[i].extend(complement)

            train_split = [list(set(train)) for train in train_split]
            train_split = train_split[self.cvfold_index]
            val_split = val_split[self.cvfold_index]

        if self.split == 'val':
            complexes_this = val_split
        elif self.split == 'train':
            complexes_this = train_split
        elif self.split == 'all':
            complexes_this = complex_list

        entries = []
        for cplx in complexes_this:
            #  single or multiple
            if self.is_single == 0:
                for complex_item in complex_to_entries[cplx]:
                    if complex_item['num_muts'] > 1:
                        continue
                    else:
                        entries += [complex_item]
            elif self.is_single == 1:
                for complex_item in complex_to_entries[cplx]:
                    if complex_item['num_muts'] == 1:
                        continue
                    else:
                        entries += [complex_item]
            else:
                entries += complex_to_entries[cplx]

        self.entries = entries
        
    def _preprocess_entries(self):
        entries_full = load_category_entries(self.csv_path, self.cath_domain_cache, self.prior_dir, self.pdb_wt_dir, self.pdb_mt_dir, self.blocklist)
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries_full, f)      # 按数据集划分
        return entries_full

    def _load_structures(self, reset):
        if not os.path.exists(self.structures_cache) or reset:
            self.structures = self._preprocess_structures(reset)
        else:
            return None

    @property
    def lmdb_path(self):
        return os.path.join(self.cache_dir, 'structures.lmdb')
    @property
    def keys_path(self):
        return os.path.join(self.cache_dir, 'keys.pkl')
    @property
    def chains_path(self):
        return os.path.join(self.cache_dir, 'chains.pkl')

    def _preprocess_structures(self, reset):
        if os.path.exists(self.lmdb_path):
            os.system(f'rm {self.lmdb_path}')

        pdbcode_list = list((e['#Pdb'],e['pdb_wt_path'], e['pdb_mt_path'], e['group_ligand'], e['group_receptor']) for e in self.entries_full)  # 这里是按结构process：wt & mt
        pdbcode_list.sort()
        tasks = []

        # generate esm2 emebddings
        if not self.esm2_650M_cache.exists():
            generate_esm2(pdbcode_list, self.esm2_650M_cache)

        for (pdbcode, pdb_wt_path, pdb_mt_path, ligand, receptor)  in tqdm(pdbcode_list, desc='Structures'):
            if not os.path.exists(pdb_wt_path):
                print(f'[WARNING] PDB not found: {pdb_wt_path}.')
                continue
            if not os.path.exists(pdb_mt_path):
                print(f'[WARNING] PDB not found: {pdb_mt_path}.')
                continue

            tasks.append(
                delayed(_process_structure)(pdb_wt_path, pdb_mt_path, self.esm2_650M_cache, ligand, receptor, pdbcode)
            )
            # _process_structure(pdb_wt_path, pdb_mt_path, self.esm2_650M_cache, ligand, receptor, pdbcode)

        # Split data into chunks
        chunk_size = 512
        task_chunks = [
            tasks[i * chunk_size:(i + 1) * chunk_size]
            for i in range(math.ceil(len(tasks) / chunk_size))
        ]

        # Establish database connection
        db_conn = lmdb.open(
            self.lmdb_path,
            map_size=self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )

        keys = []
        chains = {}
        for i, task_chunk in enumerate(task_chunks):
            with db_conn.begin(write=True, buffers=True) as txn:
                processed = Parallel(n_jobs=self.num_preprocess_jobs)(
                    task
                    for task in tqdm(task_chunk, desc=f"Chunk {i + 1}/{len(task_chunks)}")
                )
                stored = 0
                for data in processed:
                    if data is None:
                        continue
                    for key, value in data.items():
                        keys.append(key)
                        # chains["_".join(key.split("_")[1:])] = value[1]
                        txn.put(key=key.encode(), value=pickle.dumps(value))
                        stored += 1
                print(f"[INFO] {stored} processed for chunk#{i + 1}")
        db_conn.close()

        with open(self.keys_path, 'wb') as f:
            pickle.dump(keys, f)

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
        entry = self.entries[index]  # 按蛋白质复合物结构读取
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
        data_dict_wt['centrality'] = self._compute_degree_centrality(data_dict_wt)  # Cb原子
        data_dict_mt['centrality'] = self._compute_degree_centrality(data_dict_mt)  # Cb原子

        keys = {'id', 'complex_PPI', 'mutstr', 'num_muts', '#Pdb', 'ddG', 'protein_group',
                'cath_domain', 'cath_label_index', 'wt_scores_cycle', 'mut_scores_cycle'}
        for k in keys:
            data_dict_wt[k] = data_dict_mt[k] = entry[k]

        data_dict_wt['inter_energy'] = entry['inter_energy_wt']
        data_dict_mt['inter_energy'] = entry['inter_energy_mt']

        assert len(entry['mutations']) == torch.sum(data_dict_wt['aa'] != data_dict_mt['aa']),f"ID={data_dict_wt['id']},{len(entry['mutations'])},{torch.sum(data_dict_wt['aa'] != data_dict_mt['aa'])}"
        data_dict_wt['mut_flag'] = data_dict_mt['mut_flag'] = (data_dict_wt['aa'] != data_dict_mt['aa'])

        if self.transform is not None:
            data_dict_wt, _ = self.transform(data_dict_wt)
            data_dict_mt, _ = self.transform(data_dict_mt)

        return {"wt": data_dict_wt,
                "mt": data_dict_mt,
                }
def get_skempi_dataset(cfg):
    from src.utils.transforms import get_transform
    return SkempiDataset_lmdb(
        csv_path=config.data.csv_path,
        prior_dir=config.data.prior_dir,
        pdb_wt_dir=config.data.pdb_wt_dir,
        pdb_mt_dir=config.data.pdb_mt_dir,
        cache_dir=config.data.cache_dir,
        num_cvfolds=self.num_cvfolds,
        cvfold_index=fold,
        transform=get_transform(config.data.transform)
    )

if __name__ == '__main__':
    # subset = HER2
    # subset = S285
    # subset = CR6261
    # python skempi_parallel.py --reset --subset skempi_v2
    parser0 = argparse.ArgumentParser("First parser for initial arguments")
    parser0.add_argument('--subset', type=str, default=f'skempi_v2')
    # 解析已知参数并获取剩下的参数
    args0, remaining_args = parser0.parse_known_args()
    subset = args0.subset

    parser = argparse.ArgumentParser(description="Second parser for remaining arguments")
    parser.add_argument('--csv_path', type=str, default=f'../../data/SKEMPI2/{subset}_cache/{subset}.csv')
    parser.add_argument('--prior_dir', type=str, default=f'../../data/SKEMPI2/{subset}_cache')
    parser.add_argument('--pdb_wt_dir', type=str, default=f'../../data/SKEMPI2/{subset}_cache/wildtype')
    parser.add_argument('--pdb_mt_dir', type=str, default=f'../../data/SKEMPI2/{subset}_cache/optimized')
    parser.add_argument('--cache_dir', type=str, default=f'../../data/SKEMPI2/{subset}_cache/entries_cache')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args(remaining_args)
    if subset == "S285" or  subset == "HER2" or subset == "CR6261" or subset == "Demo":
        blocklist = {}
        os.system(f'python build_mutant_case.py --subset {subset}')  # generate mutant structures by FoldX
    else:
        blocklist = frozenset({'1KBH'})
        os.system('python build_mutant_skempi.py')  # generate mutant structures by FoldX

    dataset = SkempiDataset_lmdb(
        csv_path = args.csv_path,
        prior_dir = args.prior_dir,
        pdb_wt_dir = args.pdb_wt_dir,
        pdb_mt_dir=args.pdb_mt_dir,
        cache_dir = args.cache_dir,
        split = 'train',
        num_cvfolds=1,
        cvfold_index=0,
        reset=args.reset,
        blocklist=blocklist,
    )
    print(len(dataset))
