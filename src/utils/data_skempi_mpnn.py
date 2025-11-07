import math
import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate
from copy import deepcopy

DEFAULT_PAD_VALUES = {
    'aa': 21,
    'aa_masked': 21,
    'aa_true': 21,
    'chain_nb': -1, 
    'pos14': 0.0,
    'chain_id': ' ', 
    'icode': ' ',
}

class PaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, eight=True):
        super().__init__()
        # self.patch_size = patch_size
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n,f"ID={x.size(0)},{n}"
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            return x
        elif isinstance(x, np.float32):
            return torch.tensor(x).unsqueeze(-1)
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

        # return torch.cat([
        #     torch.ones([l], dtype=torch.float16),
        #     torch.zeros([n - l], dtype=torch.float16)
        # ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0]["wt"].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d["wt"].keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data["wt"][self.length_ref_key].size(0) for data in data_list])
        max_length = math.ceil(max_length / 8) * 8
        # if max_length < self.patch_size:
        #     max_length = self.patch_size

        keys = self._get_common_keys(data_list)
        
        # if self.eight:
        #     max_length = math.ceil(max_length / 8) * 8

        data_list_padded = []
        for data in data_list:
            data_dict = {}
            for flag in ["wt", "mt"]:
                data_padded = {
                    k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                    for k, v in data[flag].items()
                    if k in keys
                }
                data_padded['mask'] = self._get_pad_mask(data[flag][self.length_ref_key].size(0), max_length)
                data_dict[flag] = data_padded
            data_list_padded.append(data_dict)

        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        ddG = torch.tensor([data["wt"]["ddG"] for data in data_list],dtype=torch.float32).unsqueeze(-1)
        # ddG_FoldX = torch.tensor([data["wt"]["ddG_FoldX"] for data in data_list], dtype=torch.float32).unsqueeze(-1)
        batch['ddG'] = ddG
        # batch['ddG_FoldX'] = ddG_FoldX

        return batch

MPNN_PAD_VALUES = {
            'aa': 0,
            'chain_nb': 0,
            'residue_idx': -100,
            'mask': 0,
        }

class MPNNPaddingCollate(PaddingCollate):
    def __init__(self, length_ref_key='aa', pad_values=MPNN_PAD_VALUES):
        super().__init__(length_ref_key=length_ref_key, pad_values=pad_values)
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        max_length = math.ceil(max_length / 8) * 8

        data_list_padded = []
        num_mut_chains = []
        for data in data_list:
            mpnn_data = {
                'X': data['pos_heavyatom'][..., :4, :],
                'aa': data['aa'],
                'aa_mut': data['aa_mut'],
                'mask': torch.ones_like(data['aa']),
                'chain_M': data['mut_flag'],
                'chain_encoding_all': data['chain_nb_cycle'] + 1,
                # 'residue_idx': reset_residue_idx(data['res_nb']),
                'residue_idx': data['residue_idx'],
                'complex': data['complex'],
                'ddG': data['ddG'],
                '#Pdb': data['#Pdb'],
                'num_muts': data['num_muts'],
                'mutstr': data['mutstr'],
            }
            mpnn_data_padded = {k: self._pad_last(v, max_length, value=self._get_pad_value(k)) for k, v in mpnn_data.items()}
            data_list_padded.append(mpnn_data_padded)

            mut_indices = torch.nonzero(data['mut_flag'], as_tuple=True)[0]
            mut_chains = torch.unique(mpnn_data['chain_encoding_all'][mut_indices], dim=0)  # 突变链编号, 从1开始编号
            num_mut_chains.append(mut_chains.size(0))

            for chain_idx in mut_chains.tolist():
                chain_mask = mpnn_data_padded['chain_encoding_all'] == chain_idx
                single_chain_mpnn_data_padded = deepcopy(mpnn_data_padded)
                single_chain_mpnn_data_padded['mask'] = chain_mask
                single_chain_mpnn_data_padded['chain_M'] = chain_mask * mpnn_data_padded['chain_M']
                single_chain_mpnn_data_padded['X'] = single_chain_mpnn_data_padded['X'] * chain_mask[:, None, None]
                single_chain_mpnn_data_padded['aa'] = single_chain_mpnn_data_padded['aa'] * chain_mask
                single_chain_mpnn_data_padded['aa_mut'] = single_chain_mpnn_data_padded['aa_mut'] * chain_mask
                data_list_padded.append(single_chain_mpnn_data_padded)
        batch = default_collate(data_list_padded)
        batch['num_mut_chains'] = num_mut_chains

        return batch

