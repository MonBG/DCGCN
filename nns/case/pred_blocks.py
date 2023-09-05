"""
Seq2Seq format Prediction Blocks
"""

import numpy as np
import torch
import torch.nn as nn
from nns.case.utils_blocks import DyGCN
from utils.graph_utils import spectral_norm_tensor


class CascadeGCNv2d4(nn.Module):
    """
    Improve based on v2
    1. add pre-process linear to apply skip connection
    2. multiple layer GCN
    """
    def __init__(self, num_nodes, input_dim, output_dim, hidden_dim, dist_adj,
                 num_layers=4, seq_len=12, horizon=12, **kwargs):
        super(CascadeGCNv2d4, self).__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.horizon = horizon
        self.dist_adj = dist_adj
        if isinstance(self.dist_adj, np.ndarray):
            self.dist_adj = torch.from_numpy(self.dist_adj)
        self.dist_adj = spectral_norm_tensor(self.dist_adj)

        self.pre_fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.block_inter = nn.ModuleList([DyGCN(num_nodes, hidden_dim, hidden_dim, hidden_dim) for _ in range(self.num_layers)])
        self.block_intra = nn.ModuleList([DyGCN(num_nodes, hidden_dim, hidden_dim, hidden_dim) for _ in range(self.num_layers)])
        self.block_dist = nn.ModuleList([DyGCN(num_nodes, hidden_dim, hidden_dim, hidden_dim) for _ in range(self.num_layers)])
        self.pred_conv = nn.Conv2d(seq_len, horizon * output_dim, kernel_size=(1, 2*hidden_dim), bias=True)

    def forward(self, x, graphs):
        # x: (T, B, N, D)
        # graphs: (T, B, 2, N, N)
        T, B, N, _ = x.shape
        inter_graph, intra_graph = graphs.permute(2, 0, 1, 3, 4)  # (T, B, N, N)
        x = self.pre_fc(x)
        x1, x2 = x, x
        for i in range(self.num_layers):
            x1 = self.block_inter[i](x1, inter_graph) + x1
        for i in range(self.num_layers):
            x1 = self.block_intra[i](x1, intra_graph) + x1
        for i in range(self.num_layers):
            x2 = self.block_dist[i](x2, self.dist_adj.repeat(T, B, 1, 1).to(x.device)) + x2
        x = torch.concat([x1, x2], dim=-1)
        out = self.pred_conv(x.permute(1, 0, 2, 3)).squeeze(-1)  # (B, T*F, N)
        out = out.reshape(B, self.horizon, -1, self.num_nodes).permute(1, 0, 3, 2)
        return out








