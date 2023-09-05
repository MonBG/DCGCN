import torch
import torch.nn as nn
import torch.nn.functional as F
from nns.basic_blocks import BatchNormX, GraphConvolution


class DistGraphEncoding(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, dist_adj, num_layers=3):
        super(DistGraphEncoding, self).__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.dist_adj = dist_adj  # (N, N)

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(DyGCN(num_nodes, input_dim, hidden_dim, hidden_dim))
        for _ in range(1, self.num_layers):
            self.gcn_layers.append(DyGCN(num_nodes, hidden_dim, hidden_dim, hidden_dim))

    def forward(self, x):
        # x: (T, B, N, D)
        T, B, _, _ = x.shape
        graphs = self.dist_adj.repeat(T, B, 1, 1).to(x.device)
        feats = self.gcn_layers[0](x, graphs)
        for i in range(1, self.num_layers):
            feats = self.gcn_layers[i](feats, graphs)
        return torch.concat([x, feats], dim=-1)  # (T, B, N, D+H)


class DyGCN(nn.Module):
    def __init__(self, num_nodes, in_feats_dim, out_feats_dim, hidden_dim):
        super(DyGCN, self).__init__()

        # self.fc1 = nn.Sequential(nn.Linear(in_feats_dim, out_feats_dim), nn.ReLU())
        # self.gcn1 = GraphConvolution(out_feats_dim, hidden_dim, bias=False, skip_connect=True)
        self.fc1 = nn.Sequential(nn.Linear(in_feats_dim, hidden_dim), nn.ReLU(inplace=True))
        self.gcn1 = GraphConvolution(hidden_dim, hidden_dim, bias=False, skip_connect=True)
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, out_feats_dim), nn.ReLU(inplace=True))
        self.bn = nn.BatchNorm2d(num_nodes)

    def forward(self, x, adj):
        """
        :param x: (T, B, N, D)
        :param adj: (T, B, N, N)
        :return: (T, B, N, F)
        """
        T, B, N, _ = x.shape
        t = self.fc1(x)
        t2 = F.relu(self.gcn1(t.reshape(T*B, N, -1), adj.reshape(T*B, N, -1)), inplace=True).reshape(T, B, N, -1)
        t3 = self.fc2(t2).permute(1, 2, 0, 3)  # (B, N, T, H)
        out = self.bn(t3).permute(2, 0, 1, 3)
        return out


class MultiHeadLinear(nn.Module):
    """
    Equivalent to TimeBlock
    The design of multi-head seems meaningless
    """
    def __init__(self, input_dim, output_dim, num_heads=3):
        super(MultiHeadLinear, self).__init__()
        self.num_heads = num_heads
        self.mlps = nn.ModuleList()
        for _ in range(num_heads):
            self.mlps.append(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        # x: (T, B, N, D)
        tmp = self.mlps[0](x)
        for i in range(1, self.num_heads):
            tmp = tmp + self.mlps[i](x)
        out = F.relu(tmp)
        return out


class NodeNorm(nn.Module):
    def __init__(self, num_nodes):
        super(NodeNorm, self).__init__()
        self.bn = nn.BatchNorm2d(num_nodes)

    def forward(self, x):
        # x: (T, B, N, D)
        return self.bn(x.permute(1, 2, 0, 3)).permute(2, 0, 1, 3)
