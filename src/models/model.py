# -*- coding: utf-8 -*-
from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
from collections.abc import Sequence
import math
import pandas as pd
import copy

PI = math.pi
import torch
import random
from torch import optim
from torch.utils.data import DataLoader
import torch.utils
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
from src.modules.encoders.single import PerResidueEncoder, AAEmbedding
from src.utils.protein.dihedral_chi import CHI_PI_PERIODIC_LIST
from copy import deepcopy
from .adapter import Adapter
from torch.distributions.normal import Normal

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.relpos_embed = nn.Embedding(2 * max_relative_feature + 1, num_embeddings)
        self.chains_embed = nn.Embedding(2, num_embeddings)

    def forward(self, offset, chains):
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        E = self.relpos_embed(d) + self.chains_embed(chains)
        return E

def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.contiguous().view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

def gaussian(x, mean, std):
    pi = 3.1415926
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class AApair(nn.Module):
    def __init__(self, K=16, max_aa_types=22):
        super().__init__()
        self.K = K
        self.max_aa_types = max_aa_types
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, self.K, padding_idx=21)
    def forward(self, aa, E_idx, mask_attend):
        # Pair identities[氨基酸类型pair编码]
        aa_pair = ((aa[:, :, None] + 1) % self.max_aa_types) * self.max_aa_types + \
                  ((aa[:, None, :] + 1) % self.max_aa_types)
        aa_pair = torch.clamp(aa_pair, min=21)
        aa_pair = torch.where(aa_pair % self.max_aa_types == 0, 21, aa_pair)

        aa_pair_neighbor = torch.gather(aa_pair, 2, E_idx)
        feat_aapair = self.aa_pair_embed(aa_pair_neighbor.to(torch.long))

        return feat_aapair * mask_attend.unsqueeze(-1)


class ResiduePairEncoder(nn.Module):
    def __init__(self, edge_features, node_features, cfg, num_positional_embeddings=16, num_rbf=16):
        """ Extract protein features """
        super(ResiduePairEncoder, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k1 = cfg.k1
        self.top_k2 = cfg.k2
        self.top_k3 = cfg.k3
        self.long_range_seq = cfg.long_range_seq
        self.noise_bb = cfg.noise_bb
        self.noise_sd = cfg.noise_sd
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # node_in, edge_in = 6, num_positional_embeddings + num_rbf * 45 + 7 * 2 + 16
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 26 + 7 * 2 + 16
        self.aapair = AApair(K=16, max_aa_types=22)

        self.edge_embedding = nn.Linear(edge_in, cfg.hidden_dim, bias=False)
        self.norm_edges = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False)

        self.dropout = nn.Dropout(cfg.dropout)

    def _dist(self, X, mask_residue, residue_idx, cutoff=15.0, eps=1E-6):
        """ Pairwise euclidean distances """
        B, N = X.size(0), X.size(1)
        mask_2D = torch.unsqueeze(mask_residue, 1) * torch.unsqueeze(mask_residue, 2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = torch.sqrt(torch.sum(dX**2, 3) + eps) * mask_2D
        # mask_rball = (D < cutoff) * mask_2D
        mask_rball = mask_2D
        D_max, _ = torch.max(D, -1, keepdim=True)

        # 序列最近邻
        rel_residue = residue_idx[:, :, None] - residue_idx[:, None, :]
        mask_2D_seq = (torch.abs(rel_residue) <= (self.top_k3 - 1) / 2) * mask_rball
        D_sequence = D * mask_2D_seq  # 以距离对齐
        _, E_idx_seq = torch.topk(D_sequence, self.top_k3, dim=-1, largest=True)
        mask_seq = gather_edges(mask_2D_seq.unsqueeze(-1), E_idx_seq)[:, :, :, 0]

        # 空间最近邻
        # Identify k nearest neighbors (including self)
        D_adjust = D + (~mask_rball) * D_max
        _, E_idx_spatial = torch.topk(D_adjust, self.top_k1, dim=-1, largest=False)  # 取最小值
        # E_idx_spatial = E_idx_spatial[:,:,1:]  # ak422@163.com
        mask_spatial = gather_edges(mask_rball.unsqueeze(-1), E_idx_spatial)[:, :, :, 0]

        # 序列远 空间近
        mask_2D_seq = (torch.abs(rel_residue) <= (self.long_range_seq - 1) / 2) * mask_rball  # masked sequence
        mask_2D_Lrange = (~mask_2D_seq) * mask_rball  # 序列远
        D_adjust = D + (~mask_2D_Lrange) * D_max
        _, E_idx_Lrange = torch.topk(D_adjust, self.top_k2, dim=-1, largest=False)  # 取最小值
        mask_Lrange = gather_edges(mask_2D_Lrange.unsqueeze(-1), E_idx_Lrange)[:, :, :, 0]

        return E_idx_spatial, mask_spatial, E_idx_seq, mask_seq, E_idx_Lrange, mask_Lrange

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        # Axis of rotation
        # Replace bad rotation matrices with identity
        # I = torch.eye(3).view((1,1,1,3,3))
        # I = I.expand(*(list(R.shape[:3]) + [-1,-1]))
        # det = (
        #     R[:,:,:,0,0] * (R[:,:,:,1,1] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,1])
        #     - R[:,:,:,0,1] * (R[:,:,:,1,0] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,0])
        #     + R[:,:,:,0,2] * (R[:,:,:,1,0] * R[:,:,:,2,1] - R[:,:,:,1,1] * R[:,:,:,2,0])
        # )
        # det_mask = torch.abs(det.unsqueeze(-1).unsqueeze(-1))
        # R = det_mask * R + (1 - det_mask) * I

        # DEBUG
        # https://math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        # Columns of this are in rotation plane
        # A = R - I
        # v1, v2 = A[:,:,:,:,0], A[:,:,:,:,1]
        # axis = F.normalize(torch.cross(v1, v2), dim=-1)
        return Q
    def _orientations_coarse(self, X, E_idx, mask_attend):
        # Pair features
        u = torch.ones_like(X)
        u[:,1:,:] = X[:, 1:, :] - X[:,:-1,:]
        u = F.normalize(u, dim=-1)
        b = torch.ones_like(X)
        b[:, :-1,:] = u[:, :-1,:] - u[:, 1:,:]
        b = F.normalize(b, dim=-1)
        n = torch.ones_like(X)
        n[:,:-1,:] = torch.cross(u[:,:-1,:], u[:,1:,:], dim=-1)
        n = F.normalize(n, dim=-1)
        local_frame = torch.stack([b, n, torch.cross(b, n, dim=-1)], dim=2)
        local_frame = local_frame.view(list(local_frame.shape[:2]) + [9])

        X_neighbors = gather_nodes(X, E_idx)
        O_neighbors = gather_nodes(local_frame, E_idx)
        # Re-view as rotation matrices
        local_frame = local_frame.view(list(local_frame.shape[:2]) + [3, 3])    # Oi
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])    # Oj
        # # # Rotate into local reference frames ，计算最近邻相对x_i的局部坐标系
        t = X_neighbors - X.unsqueeze(-2)
        t = torch.matmul(local_frame.unsqueeze(2), t.unsqueeze(-1)).squeeze(-1)  # 边特征第二项
        t = F.normalize(t, dim=-1) * mask_attend.unsqueeze(-1)
        r = torch.matmul(local_frame.unsqueeze(2).transpose(-1, -2), O_neighbors)  # 边特征第三项
        r = self._quaternions(r)  * mask_attend.unsqueeze(-1)   # 边特征第三项
        t2 = (1 - 2 * t) * mask_attend.unsqueeze(-1)
        r2 = (1 - 2 * r) * mask_attend.unsqueeze(-1)

        return torch.cat([t, r, t2, r2], dim=-1)

    # def PerEdgeEncoder(self, X, E_idx, mask_attend, residue_idx, chain_labels):
    def PerEdgeEncoder(self, X, E_idx, mask_attend, residue_idx, chain_labels):
        # 1. Relative spatial encodings
        O_features = self._orientations_coarse(X, E_idx, mask_attend)

        # 2. 相对位置编码
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]
        # d_chains：链内和链间标记，为0表示链内
        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]

        E_positional = self.embeddings(offset.long(), E_chains)

        return E_positional, O_features, offset.long()

    def _set_Cb_positions(self, X, mask_atom):
        """
        Args:
            pos_atoms:  (L, A, 3)
            mask_atoms: (L, A)
        """
        Ca = X[:, :, 1]
        b = X[:, :, 1] - X[:, :, 0]
        c = X[:, :, 2] - X[:, :, 1]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1]  # 虚拟Cb原子
        X[:, :, 4] = torch.where(mask_atom[:, :, 4, None], X[:, :, 4], Cb)

        if self.training:
            X[:, :, 0:4] = X[:, :, 0:4] + self.noise_bb * torch.randn_like(X[:, :, 0:4])
            X[:, :, 4:] = X[:, :, 4:] + self.noise_sd * torch.randn_like(X[:, :, 4:])

        return X

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF
    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, batch):
        mask_atom = batch["mask_atoms"]   # N = 0; CA = 1; C = 2; O = 3; CB = 4;
        residue_idx = batch["residue_idx"]
        res_nb = batch["res_nb"]
        chain_labels = batch["chain_nb"]     # # d_chains：链内和链间标记，为1表示链内
        mask_residue = batch["mask"]
        aa = batch["aa"]

        batch["pos_heavyatom"] = self._set_Cb_positions(batch["pos_heavyatom_ori"], mask_atom)
        X = batch["pos_heavyatom"]
        N  = X[:, :, 0, :]
        Ca = X[:, :, 1, :]
        C  = X[:, :, 2, :]
        O  = X[:, :, 3, :]
        Cb = X[:, :, 4, :]

        # 这里考虑用Cb原子来计算最短距离
        E_idx_spatial, mask_spatial, E_idx_seq, mask_seq, E_idx_Lrange, mask_Lrange = self._dist(Cb, mask_residue, residue_idx)

        # spactial coding & sequential coding
        E_idx = torch.cat([E_idx_spatial, E_idx_Lrange, E_idx_seq], dim=-1)
        mask_attend = torch.cat([mask_spatial, mask_Lrange, mask_seq], dim=-1)
        mask_attend = mask_residue.unsqueeze(-1) * mask_attend
        E_positional, O_features, offset = self.PerEdgeEncoder(Cb, E_idx, mask_attend, residue_idx, chain_labels)

        # backbone rbf
        RBF_all = []
        RBF_all.append(self._get_rbf(Ca, Ca, E_idx))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O

        # sidechain rbf
        mask_heavyatom = batch["mask_heavyatom"]
        RBF_all2 = []
        for i in range(5, 15, 1):
            mask_i = mask_heavyatom[:, :, i]
            mask_j = torch.gather(mask_i.unsqueeze(-1).expand(-1, -1, mask_heavyatom.size(1)), 2, E_idx)
            RBF_all2.append(torch.sqrt(torch.sum((Cb - X[:, :, i]) ** 2, -1) + 1e-6)[:, :, None] * mask_i[:, :, None])  # [B, L, L]
        RBF_all2 = torch.cat(tuple(RBF_all2), dim=-1)
        D_A_B = torch.max(RBF_all2, dim=-1, keepdim=False)[0]
        D_A_B_neighbors = torch.gather(D_A_B[:, None, :].expand(-1, D_A_B.size(-1), -1, ), 2, E_idx)  # [B,L,K]
        RBF_all.append(self._rbf(D_A_B_neighbors))
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        aapair_feature = self.aapair(aa, E_idx, mask_attend)

        E = torch.cat((E_positional, RBF_all, O_features, aapair_feature), dim=-1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        idx_spatial = self.top_k1 + self.top_k2
        idx_seq = self.top_k3

        E_spatial = E[...,:idx_spatial,:]
        E_idx_spatial = E_idx[...,:idx_spatial]
        mask_spatial = mask_attend[...,:idx_spatial]
        E_seq = E[..., -idx_seq:, :]
        E_idx_seq = E_idx[..., -idx_seq:]
        mask_seq = mask_attend[..., -idx_seq:]

        return E_spatial, E_idx_spatial, mask_spatial, E_seq, E_idx_seq, mask_seq

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.SELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing subnetworks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)   # 对非零专家索引 按列排序,

        # print(sorted_experts.shape, index_sorted_experts.shape) # torch.Size([128, 2]) torch.Size([128, 2])
        # [[0, 2],[0, 3],[1, 4],[1, 5]] sorted_experts 将feature和experts匹配上
        # [[1, 0],[0, 1],[2, 2],[3, 3]]

        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]   # 选择batch
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for an expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape [batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero
        # expand according to batch index, so we can just split by _part_sizes

        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are no longer in log space

        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)  # weight


        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=stitched.device)
        # combine samples that have been processed by the same k experts

        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # back to log space
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MoEBlock(nn.Module):
    def __init__(self, d_model: int, args=None):
        super().__init__()

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.task_id = args.task_id
        self.noise_epsilon = 1e-2
        self.d_model = d_model
        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()
        self.apply_moe = args.apply_moe
        self.noisy_gating = True
        self.top_k = args.topk
        self.experts_num = args.experts_num # e = 22
        self.ffn_adapt = args.ffn_adapt
        self.ffn_num = args.ffn_num
        self.autorouter = args.autorouter
        self.Task_num = args.Task_num
        self.adapter_flag = args.adapter_flag

        self.adaptmlp_list = nn.ModuleList()
        if self.ffn_adapt and self.adapter_flag:
            if self.apply_moe == True:
                if self.task_id>-1: # router>1
                    self.router_list = nn.ParameterList()
                    self.w_noise_list = nn.ParameterList()
                    for i in range(args.Task_num): # Task number
                        self.router_list.append(nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True))
                        self.w_noise_list.append(nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True))
                    for i in range(self.experts_num):  #  Expert number
                        self.adaptmlp = Adapter(d_model=d_model, dropout=0.0, bottleneck=self.ffn_num,
                                                adapter_scalar=0.5,
                                                adapter_layernorm_option='none',
                                                )
                        self.adaptmlp_list.append(self.adaptmlp)
                else:  # one router for all task
                    self.router = nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True)
                    self.w_noise = nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True)
                    for i in range(self.experts_num):
                        self.adaptmlp = Adapter(d_model=d_model, dropout=0.0, bottleneck=self.ffn_num,
                                                adapter_scalar=0.5,
                                                adapter_layernorm_option='none',
                                                )
                        self.adaptmlp_list.append(self.adaptmlp)
            else:  # without moe
                self.adaptmlp = Adapter(d_model=d_model, dropout=0.0, bottleneck=self.ffn_num,
                                        adapter_scalar=0.5,
                                        adapter_layernorm_option='none',
                                        )

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
            clean_values: a `Tensor` of shape [batch_size, seq_len, n].
            noisy_values: a `Tensor` of shape [batch_size, seq_len, n].  Equal to clean values plus
              normally distributed noise with standard deviation noise_stddev.
            noise_stddev: a `Tensor` of shape [batch_size, seq_len, n], or None
            noisy_top_values: a `Tensor` of shape [batch_size, seq_len, m].
               "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
            a `Tensor` of shape [batch_size, seq_len, n].
        """
        batch_size, seq_len, n = clean_values.size()
        m = noisy_top_values.size(-1)

        # Flatten the tensors to [batch_size * seq_len, n] and [batch_size * seq_len, m]
        clean_values_flat = clean_values.view(-1, n)
        noisy_values_flat = noisy_values.view(-1, n)
        noise_stddev_flat = noise_stddev.view(-1, n) if noise_stddev is not None else None
        noisy_top_values_flat = noisy_top_values.flatten()

        # Compute threshold positions
        threshold_positions_if_in = torch.arange(batch_size * seq_len, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(noisy_top_values_flat, 0, threshold_positions_if_in), 1)  # Change shape to [batch_size * seq_len, 1]
        is_in = torch.gt(noisy_values_flat, threshold_if_in)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(noisy_top_values_flat, 0, threshold_positions_if_out), 1)

        # Compute probabilities
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values_flat - threshold_if_in) / noise_stddev_flat) if noise_stddev_flat is not None else torch.zeros_like(
            clean_values_flat)
        prob_if_out = normal.cdf((clean_values_flat - threshold_if_out) / noise_stddev_flat) if noise_stddev_flat is not None else torch.zeros_like(
            clean_values_flat)
        prob = torch.where(is_in, prob_if_in, prob_if_out)

        # Reshape prob back to [batch_size, seq_len, n]
        prob = prob.view(batch_size, seq_len, n)

        return prob

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2):
        """Noisy top-k gating.
          Args:
            x: input Tensor with shape [batch_size, seq_len, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, seq_len, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # Reshape x to [batch_size * seq_len, input_size] for linear transformation
        batch_size, seq_len, input_size = x.size()
        x_reshaped = x.view(-1, input_size)

        clean_logits = x_reshaped @ w_gate.to(x_reshaped)
        if self.noisy_gating and train:
            raw_noise_stddev = x_reshaped @ w_noise.to(x_reshaped)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # Reshape logits back to [batch_size, seq_len, num_experts]
        logits = logits.view(batch_size, seq_len, self.experts_num)

        # Calculate top-k gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=-1)
        top_k_logits = top_logits[:, :, :self.top_k]
        top_k_indices = top_indices[:, :, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        # Initialize gates tensor with zeros
        gates = torch.zeros_like(logits)

        # Scatter top-k gates into the gates tensor
        gates.scatter_(-1, top_k_indices, top_k_gates)

        # Calculate load
        if self.noisy_gating and self.top_k < self.experts_num and train:
            load = (self._prob_in_top_k(clean_logits.view(batch_size, seq_len, self.experts_num),
                                        noisy_logits.view(batch_size, seq_len, self.experts_num),
                                        noise_stddev.view(batch_size, seq_len, self.experts_num),
                                        top_logits)).sum((0, 1))
        else:
            load = self._gates_to_load(gates)

        return gates, load

    def per_sample_noisy_top_k_gating(self, x, train, router_list, w_noise_list, cath_label_index, noise_epsilon=1e-2):
        """Noisy top-k gating with per-sample router selection.
        Args:
            x: input Tensor with shape [batch_size, seq_len, input_size]
            train: a boolean - we only add noise at training time.
            router_list: list of router parameters for each task
            w_noise_list: list of noise parameters for each task
            cath_label_index: tensor of shape [batch_size, 1] indicating task ID for each sample
            noise_epsilon: a float
        Returns:
            gates: a Tensor with shape [batch_size, seq_len, num_experts]
            load: a Tensor with shape [num_experts]
        """
        batch_size, seq_len, input_size = x.size()
        x_reshaped = x.view(-1, input_size)

        # Initialize clean_logits and noise_stddev
        clean_logits = torch.zeros(batch_size * seq_len, self.experts_num, device=x.device)
        noise_stddev = torch.zeros(batch_size * seq_len, self.experts_num, device=x.device)

        # Process each unique task in the batch
        unique_tasks = torch.unique(cath_label_index)
        for task_id in unique_tasks:
            # Get indices for samples belonging to this task
            task_mask = (cath_label_index == task_id).squeeze(-1)
            task_indices = torch.where(task_mask)[0]

            if task_indices.numel() == 0:
                continue

            # Get corresponding router and noise parameters
            w_gate = router_list[task_id]
            w_noise = w_noise_list[task_id]

            # Calculate logits for this task's samples
            task_x = x[task_mask].view(-1, input_size)
            task_clean_logits = task_x @ w_gate.to(task_x)
            clean_logits[task_mask.repeat_interleave(seq_len)] = task_clean_logits

            if self.noisy_gating and train:
                task_raw_noise_stddev = task_x @ w_noise.to(task_x)
                task_noise_stddev = (self.softplus(task_raw_noise_stddev) + noise_epsilon)
                noise_stddev[task_mask.repeat_interleave(seq_len)] = task_noise_stddev

        if self.noisy_gating and train:
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # Reshape logits back to [batch_size, seq_len, num_experts]
        logits = logits.view(batch_size, seq_len, self.experts_num)

        # Calculate top-k gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=-1)
        top_k_logits = top_logits[:, :, :self.top_k]
        top_k_indices = top_indices[:, :, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        # Initialize gates tensor with zeros
        gates = torch.zeros_like(logits)

        # Scatter top-k gates into the gates tensor
        gates.scatter_(-1, top_k_indices, top_k_gates)

        # Calculate load
        if self.noisy_gating and self.top_k < self.experts_num and train:
            load = (self._prob_in_top_k(clean_logits.view(batch_size, seq_len, self.experts_num),
                                        noisy_logits.view(batch_size, seq_len, self.experts_num),
                                        noise_stddev.view(batch_size, seq_len, self.experts_num),
                                        top_logits)).sum((0, 1))
        else:
            load = self._gates_to_load(gates)

        return gates, load

    def gshard_aux_loss(self, gates, load, alpha=1e-2):
        # load: [n_experts]  实际被选次数
        # gates: [B, S, n_experts]  原始 soft gate
        n_experts = load.size(0)
        total = load.sum().clamp(min=1)
        f = load / total  # 实际负载比例
        P = gates.mean(dim=[0, 1])  # 预测概率
        loss = alpha * n_experts * (f * P).sum()
        return loss

    def forward(self, x: torch.Tensor):
        if self.apply_moe == True:
            if self.task_id > -1:
                # Directly use x for gating
                gates, load = self.noisy_top_k_gating(x, self.training, self.router_list[self.task_id],
                                                      self.w_noise_list[self.task_id])

                # Reshape gates to [batch_size * seq_len, num_experts]
                batch_size, seq_len, num_experts = gates.size()
                gates_reshaped = gates.view(-1, num_experts)

                # Reshape x to [batch_size * seq_len, input_size]
                x_reshaped = x.view(-1, x.size(-1))

                dispatcher = SparseDispatcher(self.experts_num, gates_reshaped)
                expert_inputs = dispatcher.dispatch(x_reshaped)  # 将输入数据分发到不同的专家

                # Process expert inputs
                expert_outputs = []
                for i in range(self.experts_num):
                    if expert_inputs[i].size(0) > 0:
                        expert_output = self.adaptmlp_list[i](
                            expert_inputs[i].view(expert_inputs[i].size(0), x.size(-1)),
                            add_residual=False
                        )
                        expert_outputs.append(expert_output.view(expert_output.size(0), -1))

                # Combine expert outputs, 加权求和
                y = dispatcher.combine(expert_outputs)

                # Reshape y back to [batch_size, seq_len, input_size]
                y = y.view(batch_size, seq_len, x.size(-1))
                x = y
            else:  # one router for all task
                # Directly use x for gating
                gates, load = self.noisy_top_k_gating(x, self.training, self.router, self.w_noise)

                # Reshape gates to [batch_size * seq_len, num_experts]
                batch_size, seq_len, num_experts = gates.size()
                gates_reshaped = gates.view(-1, num_experts)

                # Reshape x to [batch_size * seq_len, input_size]
                x_reshaped = x.view(-1, x.size(-1))

                dispatcher = SparseDispatcher(self.experts_num, gates_reshaped)
                expert_inputs = dispatcher.dispatch(x_reshaped)

                # Process expert inputs
                expert_outputs = []
                for i in range(self.experts_num):
                    if expert_inputs[i].size(0) > 0:
                        expert_output = self.adaptmlp_list[i](
                            expert_inputs[i].view(expert_inputs[i].size(0), x.size(-1)),
                            add_residual=False
                        )
                        expert_outputs.append(expert_output.view(expert_outputs[i].size(0), -1))

                # Combine expert outputs
                y = dispatcher.combine(expert_outputs)

                # Reshape y back to [batch_size, seq_len, input_size]
                y = y.view(batch_size, seq_len, x.size(-1))
                x = y
        else:  # one adapter
            adapt_x = self.adaptmlp(x, add_residual=False)
            x = adapt_x
        return x

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, cfg=None, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(num_hidden, elementwise_affine=False) for i in range(3)])

        self.W1 = nn.Linear(num_hidden + num_in * 2, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden * 3, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        # ReZero is All You Need: Fast Convergence at Large Depth
        self.resweight = nn.Parameter(torch.Tensor([0]))

        self.act = torch.nn.SELU()
        self.dense = PositionWiseFeedForward(num_in, num_in * 4)
        self.MoE_blocks = MoEBlock(num_in, cfg.Moe)
        self.trans = nn.Linear(num_in * 2, num_in, bias=False)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # pre-norm
        residual = h_V
        h_V = self.maybe_layer_norm(0, h_V, before=True, after=False)
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)         # h_j(enc) || edge
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)   # h_i(enc)
        h_EV = torch.cat([h_V_expand, h_EV], -1)  # h_i(enc) || h_j(enc) || edge
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        # ReZero
        dh = dh * self.resweight
        h_V = residual + self.dropout(dh)

        # pre-norm
        residual = h_V
        h_V = self.maybe_layer_norm(1, h_V, before=True, after=False)
        # Position-wise feedforward
        dh = self.dense(h_V)                # ffn

        h_MoE = self.MoE_blocks(residual)      # MoE
        dh = self.trans(torch.cat([h_MoE, dh], dim=-1))       # ffn || MoE

        # ReZero
        dh = dh * self.resweight                        # ReZero
        h_V = residual + self.dropout(dh)               # residual + ffn || MoE

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # pre-norm
        residual = h_E
        h_E = self.maybe_layer_norm(2, h_E, before=True, after=False)
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)  # h_j || edge
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)  # h_i
        h_EV = torch.cat([h_V_expand, h_EV], -1)   # h_i || h_j || edge
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = residual + self.dropout(h_message)

        return h_V, h_E

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        return self.layer_norms[i](x)

class FusionLayer(nn.Module):
    def __init__(self, num_in, num_hidden, cfg, normalize_before=False, dropout=0.1, num_heads=None, scale=30):
        super(FusionLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

        self.layer_norms = nn.ModuleList([nn.LayerNorm(num_hidden, elementwise_affine=False) for i in range(1)])

        self.normalize_before = normalize_before

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.SELU()
        # self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_S, h_V, h_E, E_idx, mask_attend, mask_V=None):
        """ Parallel computation of full transformer layer """
        # Concatenate h_V_i to h_E_ij
        residual = h_V
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)  # h_j || h_edge
        h_EXV_out = cat_neighbors_nodes(h_V, h_ES, E_idx)  # h_j || h_j(enc) || h_edge
        h_EXV_out = mask_attend.unsqueeze(-1) * h_EXV_out

        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_S_expand = h_S.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_S_expand, h_V_expand, h_EXV_out], -1)  # h_i || h_i(enc) ||  h_j(enc) || h_j || h_edge
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        dh = torch.sum(h_message, -2) / self.scale
        h_V = residual + self.dropout(dh)
        h_V = self.maybe_layer_norm(0, h_V, before=False, after=True)
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        return self.layer_norms[i](x)

def init_params(module):
    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        if data.dim() > 1:
            nn.init.xavier_uniform_(data)

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

def init_selected_modules(model, target_names):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if  "down_proj" in name:
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                nn.init.zeros_(module.bias)
            elif "up_proj" in name:
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

class MoE_ddG_NET(nn.Module):
    def __init__(self, cfg):
        super(MoE_ddG_NET, self).__init__()
        # Hyperparameters
        self.node_features = cfg.encoder.node_feat_dim
        self.edge_features = cfg.encoder.edge_feat_dim
        self.num_encoder_layers = cfg.encoder.num_layers
        hidden_dim = cfg.hidden_dim

        self.pair_encoder = ResiduePairEncoder(self.edge_features, self.node_features, cfg=cfg)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_es = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Residue Encoding  # N, CA, C, O, CB,
        self.single_encoders = nn.ModuleList([
            PerResidueEncoder(
                feat_dim=hidden_dim,
            )
            for _ in range(2)
        ])

        self.AA_embed = nn.ModuleList([
            AAEmbedding(infeat_dim=101, feat_dim=hidden_dim)
            for _ in range(2)
        ])

        self.binding_embed = nn.ModuleList([
            nn.Embedding(
                        num_embeddings=2,
                        embedding_dim=hidden_dim,
                        padding_idx=0,
                         )
            for _ in range(2)
        ])

        self.mut_embed = nn.ModuleList([
            nn.Embedding(
                num_embeddings=2,
                embedding_dim=hidden_dim,
                padding_idx=0,
            )
            for _ in range(2)
        ])

        # Encoder layers
        self.encoder_layers_spatial = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim, cfg=cfg, dropout=cfg.dropout, scale=math.sqrt(cfg.k1 + cfg.k2 - 3) )
            for _ in range(cfg.encoder.num_layers)
        ])
        self.encoder_layers_sequential = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim, cfg=cfg, dropout=cfg.dropout, scale=math.sqrt(cfg.k3))
            for _ in range(cfg.encoder.num_layers)
        ])

        self.single_fusion = nn.Linear(hidden_dim*2, hidden_dim)

        self.fusion_layer = nn.ModuleList([
            FusionLayer(hidden_dim * 4, hidden_dim, cfg=cfg, dropout=cfg.dropout, scale=math.sqrt(cfg.k1 + cfg.k2 + cfg.k3 - 3))
            for _ in range(1)
        ])

        self.enc_centrality = nn.Parameter(torch.Tensor([0]))

        # pred
        self.ddg_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.boltzmann_scalar = nn.Parameter(torch.ones((1)))
        self.foldx_scalar = nn.Parameter(torch.ones((1)))
        self.structure_scalar = nn.Parameter(torch.ones((1)))
        self.cath_scalar = nn.Parameter(torch.ones((1)))

        # foldx_ddg
        self.foldx_ddg = nn.Sequential(
            nn.Linear(15, hidden_dim), nn.SELU(),
            nn.AlphaDropout(cfg.dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SELU(),
            nn.AlphaDropout(cfg.dropout),
            nn.Linear(hidden_dim, 1)
        )

        # cath classifier
        self.cath_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SELU(),
                nn.AlphaDropout(cfg.dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.SELU(),
                nn.Linear(hidden_dim, cfg.num_labels)
            )

        self.BCEWithLogLoss = nn.BCEWithLogitsLoss()

        self.apply(lambda module: init_params(module))
        # 只初始化名为 "down_proj" 和 "up_proj" 的模块
        target_modules = ["down_proj", "up_proj"]
        init_selected_modules(self, target_modules)  # Lora init

    def gather_centrality(self, nodes, neighbor_idx, mask):
        # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
        # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
        neighbors_flat = neighbor_idx.contiguous().view((neighbor_idx.shape[0], -1))
        # Gather and re-pack
        neighbor_features = torch.gather(nodes, 1, neighbors_flat)
        neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:2] + [-1])
        # 　增加可学习参数
        neighbor_features = neighbor_features + self.enc_centrality.unsqueeze(0)

        neighbor_sum = torch.sum(neighbor_features, dtype=torch.float16, dim=-1)
        centrality_norm = torch.zeros_like(neighbor_sum)
        batch_indices, seq_indices = mask.nonzero(as_tuple=True)
        if len(batch_indices) > 0:
            # 按样本分组处理
            for i in range(neighbor_sum.shape[0]):
                sample_mask = (batch_indices == i)
                if sample_mask.any():
                    valid_seq_indices = seq_indices[sample_mask]
                    valid_values = neighbor_sum[i][valid_seq_indices]
                    normalized_values = torch.nn.functional.normalize(valid_values.unsqueeze(0), dim=-1).squeeze(0)
                    centrality_norm[i][valid_seq_indices] = normalized_values

        return centrality_norm[:, :, None]

    def _random_flip_chi(self, chi, chi_alt, prob=0.2):
        """
        Args:
            chi: (L, 4)
            flip_prob: float
            chi_alt: (L, 4)
        """
        chi_new = torch.where(
            torch.rand_like(chi) <= prob,
            chi_alt,
            chi,
        )
        return chi_new

    def chi_mask(self, chi, batch, mask_single=1, mask_multi=4, mask_other=0):
        # mask mutation site and interface
        zero_chi = torch.zeros_like(chi)
        # mask_flag = batch['mut_flag'].float() + batch['interface_flag'].float()
        mask_flag = batch['mut_flag'].float()[:, :, None]
        if mask_other == 0:
            chi_other = chi * (1 - mask_flag)
        elif mask_other == 4:
            chi_other = zero_chi
        else:
            chi_other = torch.cat([zero_chi[:, :, -mask_other:],
                                   (chi * (1 - mask_flag))[:, :, :(4 - mask_other)]], dim=-1)

        if mask_single == 0:
            chi_single = chi * mask_flag
        elif mask_single == 4:
            chi_single = zero_chi
        else:
            chi_single = torch.cat([zero_chi[:, :, -mask_single:],
                                    (chi * mask_flag)[:, :, :(4 - mask_single)]], dim=-1)

        if mask_multi == 0:
            chi_multi = chi * mask_flag
        elif mask_multi == 4:
            chi_multi = zero_chi
        else:
            chi_multi = torch.cat([zero_chi[:, :, -mask_multi:],
                                   (chi * mask_flag)[:, :, :(4 - mask_multi)]], dim=-1)

        chi_select = chi_other + chi_single * (batch["num_muts"] == 1)[:, None, None] + \
                     chi_multi * (batch["num_muts"] > 1)[:, None, None]
        return chi_select

    def dihedral_encode(self, batch, code_idx):
        mask_residue = batch['mask']
        chi = self._random_flip_chi(batch['chi'], batch['chi_alt'])
        # chi_select = chi * (1 - batch['mut_flag'].float())[:, :, None]
        if self.training:
            chi_select = self.chi_mask(chi, batch, mask_single=3, mask_multi=3, mask_other=1)
        else:
            chi_select = self.chi_mask(chi, batch, mask_single=1, mask_multi=4, mask_other=2)

        x = self.single_encoders[code_idx](
            aa=batch['aa'],
            aa_esm2=batch['aa_esm2'],
            X=batch["pos_heavyatom"], mask_atom=batch['mask_heavyatom'],
            phi=batch['phi'], phi_mask=batch['phi_mask'],
            psi=batch['psi'], psi_mask=batch['psi_mask'],
            chi=chi_select, chi_mask=batch['chi_mask'],
            mask_residue=mask_residue
        )

        # 氨基酸极性编码
        aa_embed = self.AA_embed[code_idx](batch['aa'], mask_residue)

        # binding chains
        b = self.binding_embed[code_idx](batch["is_binding"])
        m = self.mut_embed[code_idx](batch['mut_flag'].long())

        x = x + aa_embed + b + m  # (6,128,128)

        return x

    def encode(self, batch):
        # 编码器
        E_spatial, E_idx_spatial, mask_spatial, E_seq, E_idx_seq, mask_seq= self.pair_encoder(batch)
        h_E_spatial = self.W_e(E_spatial)
        h_E_seq = self.W_es(E_seq)

        mask = batch['mask']
        # Encoder
        h_V_spatial = self.dihedral_encode(batch, 0)
        for i, layer in enumerate(self.encoder_layers_spatial):
            h_V_spatial, h_E_spatial = layer(h_V_spatial, h_E_spatial, E_idx_spatial, mask, mask_spatial)

        h_V_sequential = h_V_spatial
        for i, layer in enumerate(self.encoder_layers_sequential):
            h_V_sequential, h_E_seq = layer(h_V_sequential, h_E_seq, E_idx_seq, mask, mask_seq)

        h_S = self.dihedral_encode(batch, 1)
        h_V = self.single_fusion(torch.cat([h_V_spatial, h_V_sequential],dim=-1))
        E_idx = torch.cat([E_idx_spatial, E_idx_seq], dim=-1)
        h_E = torch.cat([h_E_spatial, h_E_seq], dim=-2)
        mask_attend = torch.cat([mask_spatial, mask_seq], dim=-1)

        for i, layer in enumerate(self.fusion_layer):
            h_V = layer(h_S, h_V, h_E, E_idx, mask_attend, mask)

        # 中心性
        h_centrality = self.gather_centrality(batch["centrality"], E_idx, mask)
        h_V = h_V * h_centrality

        return h_V

    def loss_cal(self, ddg_pred, ddg_pred_inv, ddg_true, is_single, num_single, num_multi):
        loss_single = (F.mse_loss(ddg_pred * is_single, ddg_true * is_single, reduction="sum") + F.mse_loss(ddg_pred_inv * is_single,-ddg_true * is_single,reduction="sum")) / (2 * num_single)
        loss_multi = (F.mse_loss(ddg_pred * (1 - is_single), ddg_true * (1 - is_single), reduction="sum") + F.mse_loss(
            ddg_pred_inv * (1 - is_single), -ddg_true * (1 - is_single), reduction="sum")) / (2 * num_multi)
        loss = 0.6 * loss_single + 0.4 * loss_multi
        return loss

    def forward(self, batch):
        batch_wt = batch["wt"]
        batch_mt = batch["mt"]
        B, L = batch_wt['aa'].size()
        is_single = torch.where(batch_wt["num_muts"] > 1, 0, 1)[:, None]  # True: single, False: multiple
        num_single = torch.clamp(torch.sum(batch_wt["num_muts"] == 1), 1)
        num_multi = torch.clamp(torch.sum(batch_wt["num_muts"] > 1), 1)

        h_wt = self.encode(batch_wt)
        h_mt = self.encode(batch_mt)

        h_wt = h_wt * batch_wt['mut_flag'][:, :, None]
        h_mt = h_mt * batch_mt['mut_flag'][:, :, None]
        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]

        ddg_foldx = self.foldx_ddg(batch_mt['inter_energy'] * self.foldx_scalar - batch_wt['inter_energy'] * self.foldx_scalar)
        ddg_foldx_inv = self.foldx_ddg(batch_wt['inter_energy'] * self.foldx_scalar - batch_mt['inter_energy'] * self.foldx_scalar)
        loss_foldx = self.loss_cal(ddg_foldx, ddg_foldx_inv, batch['ddG'], is_single, num_single, num_multi)

        wt_scores_cycle = batch_wt['wt_scores_cycle'] * self.boltzmann_scalar
        mut_scores_cycle = batch_mt['mut_scores_cycle'] * self.boltzmann_scalar
        loss_boltzmann = F.mse_loss(mut_scores_cycle - wt_scores_cycle, batch['ddG'])

        ddg_structure = self.ddg_readout(H_mt * self.structure_scalar - H_wt * self.structure_scalar)
        ddg_structure_inv = self.ddg_readout(H_wt * self.structure_scalar - H_mt * self.structure_scalar)
        loss_structure = self.loss_cal(ddg_structure, ddg_structure_inv, batch['ddG'], is_single, num_single, num_multi)

        # cath domain classifier
        logits_wt = self.cath_classifier(H_wt * self.cath_scalar) # {0:0,1:1,2:2,3:3,4:4,6:5}
        logits_mt = self.cath_classifier(H_mt * self.cath_scalar)
        loss_cath = (self.BCEWithLogLoss(input=logits_wt, target=batch_wt['cath_domain']) + \
                    self.BCEWithLogLoss(input=logits_mt, target=batch_mt['cath_domain']))/2

        loss_stack = {
            'loss_structure': loss_structure,
            'loss_foldx': loss_foldx,
            'loss_cath': loss_cath,
            'loss_boltzmann': loss_boltzmann,
        }

        return loss_stack
    def inference(self, batch):
        batch_wt = batch["wt"]
        batch_mt = batch["mt"]

        h_wt = self.encode(batch_wt)
        h_mt = self.encode(batch_mt)
        h_wt = h_wt * batch_wt['mut_flag'][:, :, None]
        h_mt = h_mt * batch_mt['mut_flag'][:, :, None]
        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]

        ddg_structure = self.ddg_readout(H_mt - H_wt)
        ddg_foldx = self.foldx_ddg(batch_mt['inter_energy'] - batch_wt['inter_energy'])
        ddg_boltzmann = batch_mt['mut_scores_cycle'] * self.boltzmann_scalar - batch_wt['wt_scores_cycle'] * self.boltzmann_scalar
        ddg_pred = ddg_structure + ddg_foldx + ddg_boltzmann

        out_dict = {
            'ddG_pred': ddg_pred,
            'ddG_true': batch['ddG'],
        }
        return out_dict
