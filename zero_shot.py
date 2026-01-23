import os
import copy
import argparse
import pandas as pd
import numpy as np
from pymol import cmd
from pathlib import Path
import subprocess
import torch
import torch.nn as nn
import torch.utils.tensorboard
from typing import Mapping, List, Dict, Tuple, Optional
from torch.utils.data import DataLoader, Dataset
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Selection
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from tqdm.auto import tqdm
import random
import pickle
import lmdb
import h5py
import re
from thop import profile
from colorama import Fore
from collections import defaultdict
from easydict import EasyDict
from src.utils.transforms._base import _get_CB_positions

from src.utils.misc import load_config, seed_all
from src.utils.data_skempi_mpnn import PaddingCollate
from src.utils.train_mpnn import *

from src.utils.transforms import Compose, SelectAtom, AddAtomNoise, SelectedRegionFixedSizePatch
from src.utils.protein.parsers import parse_biopython_structure
from src.models.model import MoE_ddG_NET
from src.utils.skempi_mpnn import eval_skempi_three_modes, eval_HER2_modes, eval_permutation_modes
from src.datasets.InterfaceResidues import interfaceResidues
from transformers import AutoTokenizer, EsmModel
from transformers import logging
# 隐藏所有Transformers的警告信息
logging.set_verbosity_error()

from src.utils.misc import  current_milli_time
from protein_mpnn_utils import ProteinMPNN, _scores
import multiprocessing as mp
import glob
from src.utils.data_skempi_mpnn import MPNNPaddingCollate

import warnings
warnings.filterwarnings("ignore")

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

def PDB_chainset(pdb_path, partner1, partner2):
    chain_group = list([partner1, partner2])
    complex = pdb_path.name.stem

    # cath processing
    pdb_chain_set = set()
    for chain in chain_group:
        pdb_chain = (complex.lower() + chain.upper())
        pdb_chain_set.add(pdb_chain)

    return pdb_chain_set

def load_category_entries(pdb_wt_path,  partner1, partner2, individual_list):
    complex = pdb_wt_path.name.split('.')[0]
    prior_dir = pdb_wt_path.parent
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
    group1 = partner1
    group2 = partner2
    protein_1 = 'NAN'
    protein_2 = 'NAN'

    individual_list = Path(individual_list)
    try:
        with open(individual_list, 'r', encoding='utf-8') as file:  # 使用 'r' 模式打开文件，编码通常为 utf-8
            mut_str = file.readline().strip()  # 读取第一行并去除首尾空白字符
            if mut_str.endswith(';'):  # 检查是否以分号结尾
                mut_str = mut_str[:-1]  # 去掉最后一个字符（分号）
            print(f"The mutation(s) of {complex} is: {mut_str};")
    except FileNotFoundError:
        print("mutation file not found！")
    except Exception as e:
        print("read error：", e)

    mut_list = set(mut_str.split(','))
    muts = list(map(_parse_mut, mut_list))

    if muts[0]['chain'] in group1:
        group_ligand, group_receptor = group1, group2
    else:
        group_ligand, group_receptor = group2, group1

    pdb_wt_path = Path(prior_dir, 'wild_type.pdb')
    pdb_mt_path = Path(prior_dir, 'mutant.pdb')

    if not os.path.exists(pdb_wt_path) or not os.path.exists(pdb_mt_path):
        print("pdb file not exists!")
        return

    # read Interaction Energy of foldX
    foldX_dg_wt = Path(prior_dir, f'Interaction_wild_type.txt')
    foldX_dg_mt = Path(prior_dir, f'Interaction_mutant.txt')
    with open(foldX_dg_wt, "r") as file:
        lines = file.readlines()
        for line in lines:
            if complex in line:
                inter_energy_wt = line.split()[6:15] + line.split()[17:19] + [line.split()[20]] + line.split()[23:26]
                # inter_energy_wt = line.split()[6:19] + line.split()[20:26]   # Backbone Clash is not considered
                inter_energy_wt = np.array(list(map(float, inter_energy_wt)))
                break
    with open(foldX_dg_mt, "r") as file:
        lines = file.readlines()
        for line in lines:
            if complex in line:
                inter_energy_mt = line.split()[6:15] + line.split()[17:19] + [line.split()[20]] + line.split()[23:26]
                # inter_energy_mt = line.split()[6:19] + line.split()[20:26]  # Backbone Clash is not considered
                inter_energy_mt = np.array(list(map(float, inter_energy_mt)))
                break

    entry = {
        '#Pdb': complex,
        'complex': complex + '_' + partner1 + '_' + partner2,
        'mutstr': mut_str,
        'num_muts': len(muts),
        'mut_type': len(muts) > 1,
        'protein_group': protein_1,
        'group_ligand': list(group_ligand),
        'group_receptor': list(group_receptor),
        'mutations': muts,
        'ddG': 0.0,
        # 'ddG_FoldX': np.float32(ddG_FoldX),         # ddG_FoldX: Interaction
        'cath_domain': np.array([0,1,0,1,0,1]),
        'pdb_wt_path': pdb_wt_path,
        'pdb_mt_path': pdb_mt_path,
        'inter_energy_wt': inter_energy_wt,
        'inter_energy_mt': inter_energy_mt,
    }
    entries.append(entry)
    return entries

def _get_structure(pdb_path, interface, chains, pdbcode, esm2, flag):
    parser = PDBParser(QUIET=True)
    model = parser.get_structure(None, pdb_path)[0]

    data, chains_map = parse_biopython_structure(model, interface, esm2, chains)
    _structures = {}
    _structures[flag + "_" +pdbcode] = (data, chains_map)

    return _structures

from ddg_predictor import DDGPredictor

def load_state_dict_inference(state_dict, model):
    for sd, obj in zip(state_dict['models'], model):
        model.load_state_dict(sd, strict=False)

class CaseDataset(Dataset):
    MAP_SIZE = 500 * (1024 * 1024 * 1024)  # 500GB
    def __init__(self, config, device):
        super().__init__()
        self.pdb_wt_path = Path(config.pdb_wt_path)
        self.cache_dir = Path(config.cache_dir)
        self.individual_list = config.individual_list
        self.partner1 = config.partner1
        self.partner2 = config.partner2

        self.ddg_predictor = DDGPredictor(config)
        self.ddg_predictor.load_state_dict(torch.load(config.sft_ckpt_path, map_location='cpu', weights_only=False)['models'][0], strict=False)
        # self.ddg_predictor.mpnn.load_state_dict(torch.load(config.ckpt_path, map_location='cpu', weights_only=False)['model_state_dict'], strict=False)
        self.device = device

        self.data = []
        self.db_conn = None
        self.entries_cache = Path(self.cache_dir, 'entries.pkl')
        self.interfaces_cache = Path(self.cache_dir, 'interface.json')
        self.cath_domain_cache = Path(self.cache_dir, 'cath_domain_complex.json')
        self.esm2_650M_cache = Path(self.cache_dir, 'ESM2_650M.hdf5')
        self.structures_cache = Path(self.cache_dir, 'structures.lmdb')
        self.entries = None
        self.structures = None

        self.ddG_FoldX, stderr = self.build_mutant(self.pdb_wt_path, self.cache_dir, self.individual_list, self.partner1, self.partner2)

        self._load_entries(reset=True)
        self._load_structures(reset=True)

        self.transform = Compose([
            SelectAtom(config.data.transform[0].resolution),
            AddAtomNoise(config.data.transform[1].noise_backbone,
                         config.data.transform[1].noise_sidechain),
            SelectedRegionFixedSizePatch(config.data.transform[2].select_attr, patch_size=256)
        ])

    def build_mutant(self, pdb_path_wt, output_dir, individual_list, partner1, partner2, RepairPDB=False):
        complex = pdb_path_wt.stem

        files = glob.glob(f'{pdb_path_wt.parent}/{complex}_Repair.pdb')
        if files == []:
            # run FoldX repair
            command = f"./FoldX --command RepairPDB --ionStrength 0.15 --pdb-dir {pdb_path_wt.parent} --output-dir {output_dir} --pdb {pdb_path_wt.name}"
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=9600)
            if result.returncode == 0:
                print('RepairPDB success.')
            else:
                print('RepairPDB failed.')

        # buildmodel
        command = f'./FoldX --command=BuildModel --numberOfRuns=1 --pdb={f"{complex}_Repair.pdb"}  --mutant-file={individual_list}  --output-dir={output_dir} --pdb-dir={output_dir} >{output_dir}/foldx.log'
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print('BuildModel success.')
        else:
            print('BuildModel failed.')

        # optimize for mutant
        command = f"./FoldX --command=Optimize --pdb={f'{complex}_Repair_1.pdb'}  --output-dir={output_dir} --pdb-dir={output_dir} >{output_dir}/foldx.log"
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print('Optimize mutant success.')
        else:
            print('Optimize mutant failed.')

        # AnalyseComplex for wild-type
        command = f"./FoldX --command=AnalyseComplex --pdb={f'WT_{complex}_Repair_1.pdb'} --analyseComplexChains={f'{partner1},{partner2}'} --output-dir={output_dir} --pdb-dir={output_dir} >{output_dir}/foldx.log"
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print('AnalyseComplex wild-type success.')
        else:
            print('AnalyseComplex wild-type failed.')

        # AnalyseComplex for mutant
        command = f"./FoldX --command=AnalyseComplex --pdb={f'{complex}_Repair_1.pdb'} --analyseComplexChains={f'{partner1},{partner2}'} --output-dir={output_dir} --pdb-dir={output_dir} >{output_dir}/foldx.log"
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print('AnalyseComplex mutant success.')
        else:
            print('AnalyseComplex mutant failed.')

        # Interaction energy term
        Interaction_mt = f"{output_dir}/Interaction_{complex}_Repair_1_AC.fxout"
        Interaction_wt = f"{output_dir}/Interaction_WT_{complex}_Repair_1_AC.fxout"
        with open(Interaction_mt, "r") as file:
            lines = file.readlines()
            for line in lines:
                if complex in line:
                    Inter_mt = float(line.split()[5])
        with open(Interaction_wt, "r") as file:
            lines = file.readlines()
            for line in lines:
                if complex in line:
                    Inter_wt = float(line.split()[5])

        ddG_FoldX = Inter_mt - Inter_wt
        print('ddG_FoldX = ', ddG_FoldX)

        command1 = f'mv {output_dir}/Interaction_{complex}_Repair_1_AC.fxout {output_dir}/Interaction_mutant.txt'
        command2 = f'mv {output_dir}/Interaction_WT_{complex}_Repair_1_AC.fxout {output_dir}/Interaction_wild_type.txt'
        command3 = f'mv {output_dir}/Optimized_{complex}_Repair_1.pdb {output_dir}/mutant.pdb'
        command4 = f'mv {output_dir}/WT_{complex}_Repair_1.pdb {output_dir}/wild_type.pdb'
        command5 = f'rm -r {output_dir}/*.fxout'
        command6 = f'rm -r {output_dir}/{complex}_Repair_1.pdb'
        process1 = subprocess.run(command1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process2 = subprocess.run(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process3 = subprocess.run(command3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process4 = subprocess.run(command4, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process5 = subprocess.run(command5, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process6 = subprocess.run(command6, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return (ddG_FoldX, None) if process6.returncode == 0 else (ddG_FoldX, process6.stderr)

    def save_mutations(self, mutations):
        data = pd.DataFrame(data=mutations, index=None)
        path = os.path.join(os.path.dirname(self.cache_dir), "7FAE_RBD_Fv_mutations.csv")
        data.to_csv(path)

    def clone_data(self):
        return copy.deepcopy(self.data)

    def _preprocess_entries(self):
        entries = load_category_entries(self.pdb_wt_path, self.partner1, self.partner2, self.individual_list)
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries, f)      # 按数据集划分
        return entries

    def _load_structures(self, reset=False):
        if not os.path.exists(self.structures_cache) or reset:
            self.structures = self._preprocess_structures(reset)
        else:
            return None

    def _preprocess_structures(self, reset):
        if os.path.exists(self.lmdb_path):
            os.system(f'rm {self.lmdb_path}')

        pdbcode_list = list((e['#Pdb'],e['pdb_wt_path'], e['pdb_mt_path'], e['group_ligand'], e['group_receptor']) for e in self.entries)  # 这里是按结构process：wt & mt
        pdbcode_list.sort()
        tasks = []

        # generate esm2 emebddings
        self.generate_esm2(pdbcode_list, self.esm2_650M_cache)

        for (pdbcode, pdb_wt_path, pdb_mt_path, ligand, receptor)  in pdbcode_list:
            if not os.path.exists(pdb_wt_path):
                print(f'[WARNING] PDB not found: {pdb_wt_path}.')
                continue
            if not os.path.exists(pdb_mt_path):
                print(f'[WARNING] PDB not found: {pdb_mt_path}.')
                continue

            structures = self._process_structure(pdb_wt_path, pdb_mt_path, self.esm2_650M_cache, ligand, receptor, pdbcode)

        return structures

    def generate_esm2(self, pdbcode_list, esm2_650M_cache):
        tokenizer = AutoTokenizer.from_pretrained("./data/esm2_t33_650M_UR50D")
        esm_model = EsmModel.from_pretrained("./data/esm2_t33_650M_UR50D", device_map="cpu")
        # esm_model = EsmModel.from_pretrained("./data/esm2_t33_650M_UR50D",  device_map = "auto", max_memory={0: "24GiB", 1: "2GiB", 2: "2GiB", 3: "2GiB"})
        # esm_model = EsmModel.from_pretrained("./data/esm2_t33_650M_UR50D",  device_map = "auto", max_memory={2: "40GiB"})

        with h5py.File(esm2_650M_cache, "w") as h5file:
            h5file.create_group('wt')
            h5file.create_group('mt')
            for (pdbcode, pdb_wt_path, pdb_mt_path, ligand, receptor) in pdbcode_list:
                h5file['wt'].create_group(f'{pdbcode}')
                h5file['mt'].create_group(f'{pdbcode}')
                # generate sequences
                sequences = pdb2dict(pdb_wt_path)
                for chain, sequence in sequences.items():
                    sequences_tokenized = tokenizer(sequence, return_tensors="pt")['input_ids'].squeeze(0)
                    with torch.no_grad():
                        last_hidden_states = esm_model(sequences_tokenized.unsqueeze(0)).last_hidden_state[:, 1:-1, :]
                    data_wt_chain = last_hidden_states.squeeze(0)
                    h5file['wt'][pdbcode].create_dataset(f"{chain}", data=data_wt_chain.numpy(), dtype='f')

                sequences = pdb2dict(pdb_mt_path)
                for chain, sequence in sequences.items():
                    sequences_tokenized = tokenizer(sequence, return_tensors="pt")['input_ids'].squeeze(0)
                    with torch.no_grad():
                        last_hidden_states = esm_model(sequences_tokenized.unsqueeze(0)).last_hidden_state[:, 1:-1, :]
                    data_mt_chain = last_hidden_states.squeeze(0)
                    h5file['mt'][pdbcode].create_dataset(f"{chain}", data=data_mt_chain.numpy(), dtype='f')

    def _process_structure(self, pdb_wt_path, pdb_mt_path, esm2_path, ligand, receptor, pdbcode) -> Optional[Dict]:
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
                rVal, ans = interfaceResidues('wild_type', 'c. ' + chain_A, 'c. ' + chain_B)
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
                rVal, ans = interfaceResidues('mutant', 'c. ' + chain_A, 'c. ' + chain_B)
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

    def _load_entries(self, reset=False):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries = self._preprocess_entries()    # 按数据集划分
        else:
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

    def thermo_cycle(self, batch):
        self.ddg_predictor.to(self.device)
        self.ddg_predictor.eval()

        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            wt_scores_cycle, mut_scores_cycle = self.ddg_predictor(batch)

        return wt_scores_cycle.item(), mut_scores_cycle.item()

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
        # return len(self.mutations)
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]  # 按蛋白质复合物结构读取
        pdbcode = entry['#Pdb']

        data_wt, chains_wt = self.structures["wt_" + pdbcode]  # Made a copy
        data_mt, chains_mt = self.structures["mt_" + pdbcode]  # Made a copy

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

            # BA-cycle
            if chain in entry['group_ligand']:
                data_wt[i]['chain_nb_cycle'] = torch.ones_like(data_wt[i]['chain_nb'])
                data_mt[i]['chain_nb_cycle'] = torch.ones_like(data_mt[i]['chain_nb'])
            elif chain in entry['group_receptor']:
                data_wt[i]['chain_nb_cycle'] = torch.zeros_like(data_wt[i]['chain_nb'])
                data_mt[i]['chain_nb_cycle'] = torch.zeros_like(data_mt[i]['chain_nb'])
            elif len(entry['group_ligand']) > 0 and len(entry['group_receptor']) > 0:
                continue

        # random.shuffle(chains_list)
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

        keys = {'mutstr', 'num_muts', '#Pdb', 'ddG', 'cath_domain', 'complex'}
        for k in keys:
            data_dict_wt[k] = data_dict_mt[k] = entry[k]
        data_dict_wt['inter_energy'] = entry['inter_energy_wt']
        data_dict_mt['inter_energy'] = entry['inter_energy_mt']

        assert len(entry['mutations']) == torch.sum(data_dict_wt['aa'] != data_dict_mt['aa']), f"ID={data_dict_wt['id']},{len(entry['mutations'])},{torch.sum(data_dict_wt['aa'] != data_dict_mt['aa'])}"
        data_dict_wt['mut_flag'] = data_dict_mt['mut_flag'] = (data_dict_wt['aa'] != data_dict_mt['aa'])
        data_dict_wt['aa_mut'] = data_dict_mt['aa']

        MPNN_PAD_VALUES = {
            'aa': 0,
            'aa_mut': 0,
            'chain_nb': 0,
            'residue_idx': -100,
            'mask': 0,
        }
        MPNNpadding_collate = MPNNPaddingCollate(256)
        batch = MPNNpadding_collate([data_dict_wt])

        wt_scores_cycle, mut_scores_cycle  = self.thermo_cycle(batch)
        data_dict_wt['wt_scores_cycle'] = wt_scores_cycle
        data_dict_wt['mut_scores_cycle'] = mut_scores_cycle
        data_dict_mt['wt_scores_cycle'] = wt_scores_cycle
        data_dict_mt['mut_scores_cycle'] = mut_scores_cycle

        if self.transform is not None:
            data_dict_wt, _ = self.transform(data_dict_wt)
            data_dict_mt, _ = self.transform(data_dict_mt)

        return {"wt": data_dict_wt,
                "mt": data_dict_mt,
                }

if __name__ == '__main__':
    # python zero_shot.py ./configs/inference/zero_shot.yml  --device cuda:2
    mp.set_start_method('spawn', force=True)
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
    seed_all(config.seed)

    for checkpoint in config.checkpoints:
        ckpt.append(torch.load(checkpoint, map_location=args.device))
    config_model = ckpt[0]['config']
    print(config_model)

    # load model
    cv_mgr = CrossValidation(model_factory=MoE_ddG_NET,
                             config=config_model,
                             num_cvfolds=config_model.train.num_cvfolds).to(args.device)
    # Data
    dataset = CaseDataset(
        config=config,
        device=args.device,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=PaddingCollate(config.data.batch_size),
        num_workers=args.num_workers,
    )

    results = []
    for fold in range(config_model.train.num_cvfolds):
        for i in range(len(ckpt)):
            cv_mgr.load_state_dict(ckpt[i]['model'], )
            model, _, _, _ = cv_mgr.get(fold)
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    # Prepare data
                    batch = recursive_to(batch, args.device)
                    output_dict = model.inference(batch)

                    for pdbcode, mutstr, ddg_pred in zip(batch["wt"]['#Pdb'], batch["wt"]['mutstr'], output_dict['ddG_pred']):
                        results.append({
                            'pdbcode': pdbcode,
                            'mutstr': mutstr,
                            'num_muts': len(mutstr.split(',')),
                            'ddG_pred': ddg_pred.item(),
                        })

    results = pd.DataFrame(results)
    print('ddG = ', config.ddG)
    print('ddG_pred = ', results['ddG_pred'].mean(axis=0))