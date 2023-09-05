import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BatchNormX(nn.Module):
    def __init__(self, batch_dim):
        super(BatchNormX, self).__init__()
        self.batch_dim = batch_dim
        self.batch_norm = nn.BatchNorm1d(batch_dim)

    def forward(self, x):
        D = x.shape[-1]  # D = self.batch_dim
        out = self.batch_norm(x.reshape((-1, D))).reshape(x.shape)
        return out


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_feats_dim, out_feats_dim, bias=True, skip_connect=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_feats_dim
        self.out_features = out_feats_dim
        self.skip_connect = skip_connect
        self.weight = nn.Parameter(torch.FloatTensor(in_feats_dim, out_feats_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feats_dim))
        else:
            self.register_parameter('bias', None)
        if skip_connect:
            assert in_feats_dim == out_feats_dim
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        :param input: (B, N, D)
        :param adj: (B, N, N)
        :return: (B, N, F)
        """
        support = torch.matmul(input, self.weight)  # (B, N, F)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if self.skip_connect:
            return output + input
        else:
            return output


class BatchGCN(nn.Module):
    def __init__(self, in_feats_dim, hidden_dim, out_feats_dim, dropout):
        super(BatchGCN, self).__init__()

        self.gc1 = GraphConvolution(in_feats_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, out_feats_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        :param input: (B, N, D)
        :param adj: (B, N, N)
        :return: (B, N, F)
        """
        x = F.gelu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)