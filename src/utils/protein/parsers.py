import torch
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from easydict import EasyDict
import os


from .constants import (AA, max_num_heavyatoms,
                        restype_to_heavyatom_names, 
                        BBHeavyAtom,
                        ALL_ATOM_POSNS, ALL_ATOMS)
from .icoord import get_chi_angles, get_backbone_torsions
max_num_heavyatoms = 15
max_num_allatoms = 37

def _get_residue_heavyatom_info(res: Residue):
    # 获取ALL_ATOM_POSNS的info
    pos_allatom = torch.zeros([max_num_allatoms, 3], dtype=torch.float)
    mask_allatom = torch.zeros([max_num_allatoms, ], dtype=torch.bool)
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)

    restype = AA(res.get_resname())
    for atom_name in res:
        # if atom_name == '': continue
        if  atom_name.id not in ALL_ATOMS:
            # print("exist none-standard atoms.")
            continue
        # try:
        #     idx_heavyatom = restype_to_heavyatom_names[restype].index(atom_name.id)
        # except:
        #     continue
        idx_heavyatom = restype_to_heavyatom_names[restype].index(atom_name.id)

        idx_allatom = ALL_ATOM_POSNS[atom_name.id]
        pos_heavyatom[idx_heavyatom] = torch.tensor(res[atom_name.id].get_coord().tolist(), dtype=pos_heavyatom.dtype)
        mask_heavyatom[idx_heavyatom] = True
        pos_allatom[idx_allatom] = torch.tensor(res[atom_name.id].get_coord().tolist(), dtype=pos_allatom.dtype)
        mask_allatom[idx_allatom] = True

    return pos_heavyatom, mask_heavyatom, pos_allatom, mask_allatom


def parse_pdb(path, model_id, unknown_threshold=1.0):
    parser = PDBParser()
    structure = parser.get_structure(None, path)
    return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)


def parse_mmcif_assembly(path, model_id, assembly_id=0, unknown_threshold=1.0):
    parser = MMCIFParser()
    structure = parser.get_structure(None, path)
    mmcif_dict = parser._mmcif_dict
    if '_pdbx_struct_assembly_gen.asym_id_list' not in mmcif_dict:
        return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)
    else:
        assemblies = [tuple(chains.split(',')) for chains in mmcif_dict['_pdbx_struct_assembly_gen.asym_id_list']]
        label_to_auth = {}
        for label_asym_id, auth_asym_id in zip(mmcif_dict['_atom_site.label_asym_id'], mmcif_dict['_atom_site.auth_asym_id']):
            label_to_auth[label_asym_id] = auth_asym_id
        model_real = list({structure[model_id][label_to_auth[ch]] for ch in assemblies[assembly_id]})
        return parse_biopython_structure(model_real)

def parse_biopython_structure(entity, interface_flag, esm2, chains_ordered, unknown_threshold=1.0):
    chains = Selection.unfold_entities(entity, 'C')
    # delete invalid chain
    for i, chain in enumerate(chains):
        if chain.id == " ":
            del chains[i]
    index_dict = {k: i for i, k in enumerate(chains_ordered)}
    chains = sorted(chains, key=lambda x: index_dict[x])  # keep wt and mt being consistent

    data_chains_dict = {}
    tensor_types = {
        'chain_nb': torch.LongTensor,
        'resseq': torch.LongTensor,
        'res_nb': torch.LongTensor,
        'residue_idx': torch.LongTensor,
        'seq_idx': torch.LongTensor,
        'aa': torch.LongTensor,
        'aa_esm2': torch.FloatTensor,
        'pos_heavyatom': torch.stack,
        'mask_heavyatom': torch.stack,
        'pos_allatom': torch.stack,
        'mask_allatom': torch.stack,
        'interface_flag': torch.BoolTensor,

        'phi': torch.FloatTensor,
        'phi_mask': torch.BoolTensor,
        'psi': torch.FloatTensor,
        'psi_mask': torch.BoolTensor,
        'omega': torch.FloatTensor,
        'omega_mask': torch.BoolTensor,

        'chi': torch.stack,
        'chi_alt': torch.stack,
        'chi_mask': torch.stack,
        'chi_complete': torch.BoolTensor,
    }

    count_unk = 0
    c = 1
    l0= 0
    # chain_nb = 0
    chains_list = []
    for i, chain in enumerate(chains):
        chain_nb = i
        if chain.get_id() == ' ':
            continue
        chains_list.append(chain.get_id())
        count_aa = 0
        data = EasyDict({
            'chain_nb': [],
            'resseq': [], 'res_nb': [], 'residue_idx': [], 'seq_idx': [],
            'aa': [], 'aa_esm2': [],
            'pos_heavyatom': [], 'mask_heavyatom': [],
            'pos_allatom': [], 'mask_allatom': [], 'interface_flag': [],
            'phi': [], 'phi_mask': [],
            'psi': [], 'psi_mask': [],
            'omega': [], 'omega_mask': [],
            'chi': [], 'chi_alt': [], 'chi_mask': [], 'chi_complete': [],
        })

        chain.atom_to_internal_coordinates()
        seq_this = 0   # Renumbering residues
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))   # Sort residues by resseq-icode
        for _, res in enumerate(residues):
            resname = res.get_resname()
            if not AA.is_aa(resname):
                continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')):
                continue
            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK: 
                count_unk += 1
                continue

            # Chain info
            # data.chain_id.append(chain.get_id())
            data.chain_nb.append(chain_nb)

            # Residue types
            data.aa.append(restype) # Will be automatically cast to torch.long
            data.aa_esm2.extend(AA.three2one(resname))

            # Heavy atoms
            pos_heavyatom, mask_heavyatom, pos_allatom, mask_allatom = _get_residue_heavyatom_info(res)
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)
            data.pos_allatom.append(pos_allatom)
            data.mask_allatom.append(mask_allatom)

            # Backbone torsions
            phi, psi, omega = get_backbone_torsions(res)
            if phi is None:
                data.phi.append(0.0)
                data.phi_mask.append(False)
            else:
                data.phi.append(phi)
                data.phi_mask.append(True)
            if psi is None:
                data.psi.append(0.0)
                data.psi_mask.append(False)
            else:
                data.psi.append(psi)
                data.psi_mask.append(True)
            if omega is None:
                data.omega.append(0.0)
                data.omega_mask.append(False)
            else:
                data.omega.append(omega)
                data.omega_mask.append(True)

            # Chi
            chi, chi_alt, chi_mask, chi_complete = get_chi_angles(restype, res)
            data.chi.append(chi)
            data.chi_alt.append(chi_alt)
            data.chi_mask.append(chi_mask)
            data.chi_complete.append(chi_complete)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
                seq_idx_this = 1   # 每条链从1开始计数, 序号间隔为1
            else:
                d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)
                seq_idx_this += 1

            data.resseq.append(resseq_this)         # PDB序列号
            data.res_nb.append(seq_this)            # 重新编号的序列号, 每条链从1开始
            data.residue_idx.append(100 * (c - 1) + l0 + seq_this)   #
            data.seq_idx.append(100 * (c - 1) + l0 + seq_idx_this)  # 链间隔100的PDB序号

            # interface_flag
            if chain.get_id()+'_'+str(seq_this) in interface_flag:
                data.interface_flag.append(True)
            else:
                data.interface_flag.append(False)

        l0 += seq_this
        c += 1
        if len(data.aa) == 0:
            continue
        else:
            data_chains_dict[chain_nb] = data
        # chain_nb += 1

    for _, data in data_chains_dict.items():
        if len(data.aa) == 0:
            return None, None

    if (count_unk / l0) >= unknown_threshold:
        print("UNKNOWN: ", count_unk)
        return None, None

    for i, data in data_chains_dict.items():
        for key, convert_fn in tensor_types.items():
            if key == "aa_esm2":
                data_chains_dict[i][key] = convert_fn(esm2[chains_list[i]][:])
            else:
                data_chains_dict[i][key] = convert_fn(data[key])

    return data_chains_dict, chains_list
