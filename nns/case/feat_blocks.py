import torch
import torch.nn as nn
import numpy as np
from nns.case.utils_blocks import DyGCN, NodeNorm
from utils.graph_utils import spectral_norm_tensor


def feat_embed_block(num_nodes, in_feats_dim, hidden_dim, num_layers=3,
                     dist_adj=None, agg_feats='cat', node_norm=True):
    """
    :param num_nodes:
    :param in_feats_dim:
    :param hidden_dim:
    :param num_layers:
    :param dist_adj:
    :param agg_feats: ori, withoud feats from dist; cat, concat; sum
    :param node_norm:
    :return:
    """
    # output: x: (T, B, N, hidden_dim)
    if agg_feats == 'cat':
        assert hidden_dim % 2 == 0
        hidden_dim = hidden_dim // 2

    blocks = [nn.Linear(in_feats_dim, hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(1, num_layers):
        blocks.append(nn.Linear(hidden_dim, hidden_dim))
        blocks.append(nn.ReLU(inplace=True))
    if agg_feats != "ori" and dist_adj is not None:
        blocks.append(StaticGraphEmbedding(num_nodes, hidden_dim, hidden_dim, dist_adj,
                                           num_layers=num_layers, agg_feats=agg_feats))
    if node_norm:
        blocks.append(NodeNorm(num_nodes))
    return nn.Sequential(*blocks)


class StaticGraphEmbedding(nn.Module):
    # spectral-based graph convolution for un-directed graph
    def __init__(self, num_nodes, input_dim, output_dim, adj, num_layers=3, agg_feats='cat'):
        super(StaticGraphEmbedding, self).__init__()
        self.adj = adj  # (N, N)
        self.num_layers = num_layers
        self.agg_feats = agg_feats

        if isinstance(self.adj, np.ndarray):
            self.adj = torch.from_numpy(self.adj)
        self.adj = spectral_norm_tensor(self.adj)
        self.gcn_blocks = nn.ModuleList()
        self.gcn_blocks.append(DyGCN(num_nodes, input_dim, output_dim, output_dim))
        for _ in range(1, num_layers):
            self.gcn_blocks.append(DyGCN(num_nodes, output_dim, output_dim, output_dim))

    def forward(self, x):
        # x: (T, B, N, D)
        # out: (T, B, N, F)
        T, B, _, _ = x.shape
        adjs = self.adj.to(x.device).repeat(T, B, 1, 1)
        feats = self.gcn_blocks[0](x, adjs)
        for i in range(3):
            feats = self.gcn_blocks[i](feats, adjs)
        if self.agg_feats == 'cat':
            out = torch.concat([x, feats], dim=-1)  # intput_dim + output_dim
        else:  # sum
            out = x + feats
        return out
