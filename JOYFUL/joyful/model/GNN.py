import torch.nn as nn
import torch
from torch_geometric.nn import RGCNConv, TransformerConv
import copy
import numpy as np
import torch.nn.functional as F
from GCL.models import DualBranchContrast
import GCL.losses as L

torch.cuda.manual_seed(24)

def sim(h1, h2):
    z1 = nn.functional.normalize(h1, dim=-1, p=2)
    z2 = nn.functional.normalize(h2, dim=-1, p=2)
    contrast_model = DualBranchContrast(
        loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True
    ).to(h1.device)
    loss = contrast_model(z1, z2)
    return loss

def contrastive_loss_wo_cross_network(h1, h2, ho):
    intra1 = sim(ho, h1)
    intra2 = sim(ho, h2)
    return intra1+intra2

def random_feature_mask(input_feature, drop_percent, device=None):
    if device is None:
        device = input_feature.device
    p = torch.ones(input_feature.shape, dtype=torch.float, device=device).bernoulli_(
        1 - drop_percent
    )
    aug_feature = input_feature * p
    return aug_feature

def random_edge_pert(edge_index, num_nodes, pert_percent, device=None):
    if device is None:
        device = edge_index.device
    num_edges = edge_index.shape[1]
    pert_num_edges = int(num_edges*pert_percent)
    pert_idxs = np.random.choice(num_edges, pert_num_edges, replace=False)
    edge_index[1, pert_idxs] = torch.as_tensor(
        np.random.randint(0, num_nodes, pert_num_edges), dtype=torch.long, device=device
    )
    return edge_index


def global_proximity_edge(edge_index, node_features, topk=3):
    if topk <= 0 or node_features.size(0) < 2:
        return edge_index
    k = min(topk, node_features.size(0) - 1)
    z = F.normalize(node_features, p=2, dim=-1)
    sim = torch.matmul(z, z.t())
    sim.fill_diagonal_(-1)
    neighbors = torch.topk(sim, k=k, dim=1).indices
    src = torch.arange(node_features.size(0), device=node_features.device).unsqueeze(1).repeat(1, k).reshape(-1)
    dst = neighbors.reshape(-1)
    gp_edges = torch.stack([src, dst], dim=0)
    return torch.cat([edge_index, gp_edges], dim=1)

class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GNN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2

        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations)

        self.transform_conv = TransformerConv(h1_dim, h2_dim, heads=args.gnn_nheads, concat=True)

        self.bn = nn.BatchNorm1d(h2_dim * args.gnn_nheads)
        self.device = torch.device(args.device)
        self.disable_gcl = getattr(args, "disable_gcl", False)
        self.augment_view1 = getattr(args, "augment_view1", "fm+ep")
        self.augment_view2 = getattr(args, "augment_view2", "fm+gp")
        self.fm_drop_rate = getattr(args, "fm_drop_rate", 0.25)
        self.ep_perturb_rate = getattr(args, "ep_perturb_rate", 0.10)
        self.gp_topk = getattr(args, "gp_topk", 3)
        self.cl_tau = getattr(args, "cl_tau", 0.2)

    def _sim(self, h1, h2):
        z1 = nn.functional.normalize(h1, dim=-1, p=2)
        z2 = nn.functional.normalize(h2, dim=-1, p=2)
        contrast_model = DualBranchContrast(
            loss=L.InfoNCE(tau=self.cl_tau),
            mode="L2L",
            intraview_negs=True,
        ).to(self.device)
        return contrast_model(z1, z2)

    def _contrastive_loss(self, h1, h2, ho):
        return self._sim(ho, h1) + self._sim(ho, h2)

    def _apply_aug(self, node_features, edge_index, aug_type):
        aug_embedding = node_features
        aug_edge_index = edge_index.clone()
        if "fm" in aug_type:
            aug_embedding = random_feature_mask(aug_embedding, self.fm_drop_rate, device=self.device)
        if "ep" in aug_type:
            aug_edge_index = random_edge_pert(
                aug_edge_index, node_features.shape[0], self.ep_perturb_rate, device=self.device
            )
        if "gp" in aug_type:
            aug_edge_index = global_proximity_edge(aug_edge_index, aug_embedding, self.gp_topk)
        return aug_embedding, aug_edge_index

    def forward(self, node_features, edge_index, edge_type, trainW):

        if trainW and (not self.disable_gcl):
            aug1_embedding, aug1_edge_index = self._apply_aug(node_features, edge_index, self.augment_view1)
            aug2_embedding, aug2_edge_index = self._apply_aug(node_features, edge_index, self.augment_view2)
            h1 = self.conv1(aug1_embedding, aug1_edge_index, edge_type)
            h2 = self.conv1(aug2_embedding, aug2_edge_index, edge_type)
            ho = self.conv1(node_features, edge_index, edge_type)
            loss = self._contrastive_loss(h1, h2, ho)
            x = nn.functional.leaky_relu(self.bn(self.transform_conv(ho, edge_index)))
            return x, loss
        else:
            ho = self.conv1(node_features, edge_index, edge_type)
            x = nn.functional.leaky_relu(self.bn(self.transform_conv(ho, edge_index)))
            return x, 0


