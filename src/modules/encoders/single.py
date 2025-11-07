import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.common.layers import AngularEncoding
from src.utils.protein.constants import AA1LetterCode

class AAEmbedding(nn.Module):
    def __init__(self, infeat_dim, feat_dim):
        super(AAEmbedding, self).__init__()
        self.hydropathy = {'-': 0, '#': 0, "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "W": -0.9, "G": -0.4, "T": -0.7, "S": -0.8, "Y": -1.3, "P": -1.6, "H": -3.2, "N": -3.5, "D": -3.5, "Q": -3.5, "E": -3.5, "K": -3.9, "R": -4.5}
        self.volume = {'-': 0, '#': 0, "G": 60.1, "A": 88.6, "S": 89.0, "C": 108.5, "D": 111.1, "P": 112.7, "N": 114.1, "T": 116.1, "E": 138.4, "V": 140.0, "Q": 143.8, "H": 153.2, "M": 162.9, "I": 166.7, "L": 166.7, "K": 168.6, "R": 173.4, "F": 189.9, "Y": 193.6, "W": 227.8}
        self.charge = {**{'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1}, **{x: 0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#-'}}
        self.polarity = {**{x: 1 for x in 'RNDQEHKSTY'}, **{x: 0 for x in "ACGILMFPWV#-"}}
        self.acceptor = {**{x: 1 for x in 'DENQHSTY'}, **{x: 0 for x in "RKWACGILMFPV#-"}}
        self.donor = {**{x: 1 for x in 'RKWNQHSTY'}, **{x: 0 for x in "DEACGILMFPV#-"}}

        # alphabet = 'ACDEFGHIKLMNPQRSTVWY#-'
        alphabet = AA1LetterCode + ['-']  # padding

        # self.embedding = torch.tensor([
        #     [self.hydropathy[alphabet[i]], self.volume[alphabet[i]] / 100, self.charge[alphabet[i]],
        #      self.polarity[alphabet[i]], self.acceptor[alphabet[i]], self.donor[alphabet[i]]]
        #     for i in range(len(alphabet))
        # ])
        self.register_buffer('embedding',
            torch.tensor([
                    [self.hydropathy[alphabet[i]], self.volume[alphabet[i]] / 100, self.charge[alphabet[i]],
                     self.polarity[alphabet[i]], self.acceptor[alphabet[i]], self.donor[alphabet[i]]]
                    for i in range(len(alphabet))
            ]))

        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.SELU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.SELU(),
            nn.Linear(feat_dim, feat_dim), nn.SELU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def to_rbf(self, D, D_min, D_max, stride):
        D_count = int((D_max - D_min) / stride)
        D_mu = torch.linspace(D_min, D_max, D_count)
        D_mu = D_mu.view(1, 1, -1)  # [1, 1, K]
        D_expand = torch.unsqueeze(D, -1)  # [B, N, 1]
        return torch.exp(-((D_expand - D_mu) / stride) ** 2)

    def to_rbf_(self, D, D_min, D_max, stride):
        D_count = int((D_max - D_min) / stride)
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view(1, -1)  # [1, 1, K]
        D_expand = torch.unsqueeze(D, -1)  # [B, N, 1]
        return torch.exp(-((D_expand - D_mu) / stride) ** 2)

    def transform(self, aa_vecs):
        return torch.cat([
            self.to_rbf_(aa_vecs[:, :, 0], -4.5, 4.5, 0.1),
            # self.to_rbf_(aa_vecs[:, :, 1], 0, 2.2, 0.1),
            self.to_rbf_(aa_vecs[:, :, 2], -1.0, 1.0, 0.25),
            torch.sigmoid(aa_vecs[:, :, 3:] * 6 - 3),
        ], dim=-1)

    def dim(self):
        return 90 + 22 + 8 + 3

    def forward(self, x, mask, raw=False):
        B, N = x.size(0), x.size(1)
        aa_vecs = self.embedding[x.view(-1)].view(B, N, -1)
        rbf_vecs = self.transform(aa_vecs).to(x.device)
        return aa_vecs * mask[:,:,None] if raw else self.mlp(rbf_vecs) * mask[:,:,None]

    def soft_forward(self, x):
        aa_vecs = torch.matmul(x, self.embedding)
        rbf_vecs = torch.cat([
            self.to_rbf_(aa_vecs[:, 0], -4.5, 4.5, 0.1),
            self.to_rbf_(aa_vecs[:, 1], 0, 2.2, 0.1),
            self.to_rbf_(aa_vecs[:, 2], -1.0, 1.0, 0.25),
            torch.sigmoid(aa_vecs[:, 3:] * 6 - 3),
        ], dim=-1)
        return rbf_vecs


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.act = nn.SELU()
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x

class PerResidueEncoder(nn.Module):
    def __init__(self, feat_dim, max_aa_types=22):
        super().__init__()
        self.max_aa_types = max_aa_types
        self.feat_dim = feat_dim

        self.esm_embed = NonLinear(input=1280, output_size=feat_dim)
        # self.aatype_embed = nn.Embedding(max_aa_types, feat_dim, padding_idx=21)
        # Phi, Psi, Chi1-4
        self.dihed_embed = AngularEncoding()
        infeat_dim = feat_dim + 8 * 11 + self.dihed_embed.get_out_dim(6)
        # infeat_dim = feat_dim + 8 * 65 + self.dihed_embed.get_out_dim(6)

        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.SELU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.SELU(),
            nn.Linear(feat_dim, feat_dim), nn.SELU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def _rbf_residue(self, D, num_rbf=8):
        device = D.device
        D_min, D_max, D_count = 0.2, 6.2, num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf_residue(self, A, B):
        D_A_B = torch.sqrt(torch.sum((A[:,:,:] - B[:,:,:])**2,-1) + 1e-6) #[B, L, L]
        RBF_A_B = self._rbf_residue(D_A_B)
        return RBF_A_B

    def forward(self, aa, aa_esm2, X, mask_atom,
                phi, phi_mask, psi, psi_mask, chi, chi_mask, mask_residue, gt_mask=None, pretrain=-1):
        """
        Args:
            aa_type: (B, L, 1280)
            phi, phi_mask: (B, L)
            psi, psi_mask: (B, L)
            chi, chi_mask: (B, L, 4)
            mask_residue: (B, L)  # CA残基的mask
        """
        B, L, _ = aa_esm2.size()

        # residue distance
        N = X[:, :, 0, :]
        Ca = X[:, :, 1, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]
        Cb = X[:, :, 4, :]
        RBF_all = []
        RBF_all2 = []
        RBF_all.append(self._get_rbf_residue(Ca, N))  # Ca-N
        RBF_all.append(self._get_rbf_residue(Ca, C))  # Ca-C
        RBF_all.append(self._get_rbf_residue(Ca, O))  # Ca-O
        RBF_all.append(self._get_rbf_residue(Ca, Cb))  # Ca-Cb
        RBF_all.append(self._get_rbf_residue(N, C))  # N-C
        RBF_all.append(self._get_rbf_residue(N, O))  # N-O
        RBF_all.append(self._get_rbf_residue(N, Cb))  # N-Cb
        RBF_all.append(self._get_rbf_residue(Cb, C))  # Cb-C
        RBF_all.append(self._get_rbf_residue(Cb, O))  # Cb-O
        RBF_all.append(self._get_rbf_residue(O, C))  # O-C

        for i in range(4, 15-1, 1):
            for j in range(i+1, 15, 1):
                mask_ij = mask_atom[:, :, i] * mask_atom[:, :, j]
                RBF_all2.append((torch.sqrt(torch.sum((X[:, :, i, :] - X[:, :, j, :]) ** 2, -1) + 1e-6) * mask_ij)[:, :, None])
        RBF_all2 = torch.cat(tuple(RBF_all2), dim=-1) * mask_residue[:, :, None]
        RBF_all.append(self._rbf_residue(torch.max(RBF_all2, dim=-1, keepdim=False)[0]))

        RBF_all = torch.cat(tuple(RBF_all), dim=-1) * mask_residue[:, :, None]

        aa_feat = self.esm_embed(aa_esm2) * mask_residue[:, :, None]
        # aa_feat = self.aatype_embed(aa) * mask_residue[:, :, None]

        # Disobind: interaction block
        # z1 = torch.abs(aa_feat.unsqueeze(1) - aa_feat.unsqueeze(2))
        # z2 = aa_feat.unsqueeze(1) * aa_feat.unsqueeze(2)
        # interaction_tensor = torch.cat([z1, z2], dim=-1)
        # I1 = torch.mean(interaction_tensor, dim=1)
        # I2 = torch.mean(interaction_tensor, dim=2)

        # Dihedral features [骨架二面角和侧链二面角]
        dihedral = torch.cat(
            [phi[..., None], psi[..., None], chi],
            dim=-1
        ) # (B, L, 6)
        dihedral_mask = torch.cat([
            phi_mask[..., None], psi_mask[..., None], chi_mask],
            dim=-1
        ) # (B, L, 6)
        dihedral_feat = self.dihed_embed(dihedral[..., None]) * dihedral_mask[..., None] # (B, L, 7, feat)
        dihedral_feat = dihedral_feat.reshape(B, L, -1)

        input_dihedral_feat = dihedral_feat * mask_residue[:, :, None]
        residue_feat = torch.cat([aa_feat, input_dihedral_feat, RBF_all], dim=-1)

        # Mix
        out_feat = self.mlp(residue_feat) # (B, L, F)
        out_feat = out_feat * mask_residue[:, :, None]

        return out_feat