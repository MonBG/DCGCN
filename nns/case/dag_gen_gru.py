import math

import torch
import torch.nn as nn
from nns.case.mat_gru import MatGRU
from nns.case.dag_gen_block import DGBlockV1, DAGGumbelSigmoid
from nns.case.sem_block import MaskGCNSEM
from nns.post_processing_blocks import PostProcessingBlocV2
from nns.case.feat_blocks import feat_embed_block
from utils.graph_utils import spatial_norm_tensor


class DagGenGRUg4s2v1(nn.Module):
    """ Causal DAG generator
    - pixel-wise version of DagGenGRUv2
    - modeling spatial dependency by dot-product between nodes (similarity measurement)
    - modeling temporal dependency with GRU
    - generate graph with MLP
    - intra-slice post process
    - graph sparsity loss
    """

    def __init__(self, num_nodes, in_feats_dim, out_feats_dim, hidden_dim, num_layers=2, num_heads=4,
                 feats_layers=3, dist_adj=None, agg_feats='ori', node_norm=False, use_norm=False,
                 use_pp=False, step_pri=0.01, step_dual=0.01, reg_sp_intra=2e-3, num_intra_pp_iters=1000, **kwargs):
        super(DagGenGRUg4s2v1, self).__init__()
        GRU_FOLD = 4
        self.num_nodes = num_nodes
        self.in_feats_dim = in_feats_dim
        self.out_feats_dim = out_feats_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_norm = use_norm
        self.use_pp = use_pp

        assert hidden_dim % num_heads == 0

        self.embed = feat_embed_block(num_nodes, in_feats_dim, hidden_dim, num_layers=feats_layers,
                                      dist_adj=dist_adj, agg_feats=agg_feats, node_norm=node_norm)
        self.W_inter_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_inter_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_intra_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_intra_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.graph_gru_inter = MatGRU(num_nodes ** 2, num_heads, num_heads * GRU_FOLD, num_layers)
        self.graph_gru_intra = MatGRU(num_nodes ** 2, num_heads, num_heads * GRU_FOLD, num_layers)
        self.graph_gen_inter = DGBlockV1(num_heads * GRU_FOLD, hidden_dim, 1)
        self.graph_gen_intra = DGBlockV1(num_heads * GRU_FOLD, hidden_dim, 1)
        self.gumbel_sigmoid = DAGGumbelSigmoid()
        if self.use_pp:
            self.intra_pp = PostProcessingBlocV2(step_pri, step_dual, reg_sp_intra, num_intra_pp_iters)
        self.sem_encoder = MaskGCNSEM(num_nodes, hidden_dim, out_feats_dim, hidden_dim)

    def split_heads(self, x):
        # x: (T, B, N, H)
        T, B, N, _ = x.shape
        x = x.reshape(T, B, N, self.num_heads, -1)
        x = x.permute(0, 1, 3, 2, 4)  # (T, B, heads, N, H // heads)
        return x

    @staticmethod
    def flat_pixels(x):
        T, B, C, N, _ = x.shape
        x = x.reshape(T, B, C, -1).transpose(-1, -2)  # (T, B, N^2, C)
        return x

    @staticmethod
    def unflat_pixels(x):
        T, B, N2, C = x.shape
        N = int(math.sqrt(N2))
        x = x.reshape(T, B, N, N, C).permute(0, 1, 4, 2, 3)  # (T, B, C, N, N)
        return x

    def forward(self, x_with_pre, gen_graph_only=False):
        # x_with_pre: (T+1, B, N, D)
        x_with_pre = self.embed(x_with_pre)  # (T+1, B, N, H)
        pre_x, x = x_with_pre[:-1, ...], x_with_pre[1:, ...]

        inter_q = self.split_heads(self.W_inter_q(x))
        inter_k = self.split_heads(self.W_inter_k(pre_x))
        intra_q = self.split_heads(self.W_intra_q(x))
        intra_k = self.split_heads(self.W_intra_k(x))

        pre_feats = self.flat_pixels(torch.matmul(inter_q, inter_k.transpose(-1, -2)))  # (T, B, N^2, heads)
        cur_feats = self.flat_pixels(torch.matmul(intra_q, intra_k.transpose(-1, -2)))  # (T, B, N^2, heads)
        hiddens_inter, _ = self.graph_gru_inter(pre_feats)  # (T, B, N^2, heads * GRU_FOLD)
        hiddens_intra, _ = self.graph_gru_intra(cur_feats)  # (T, B, N^2, heads * GRU_FOLD)
        inter_graph = self.unflat_pixels(self.graph_gen_inter(hiddens_inter)).squeeze(dim=2)  # (T, B, N, N)
        intra_graph = self.unflat_pixels(self.graph_gen_intra(hiddens_intra)).squeeze(dim=2)  # (T, B, N, N)
        assert torch.isnan(inter_graph).sum() == 0
        assert torch.isnan(intra_graph).sum() == 0
        inter_graph = self.gumbel_sigmoid(inter_graph)
        intra_graph = self.gumbel_sigmoid(intra_graph, mask=True)
        if self.use_pp:
            intra_graph = self.intra_pp(intra_graph.reshape(-1, self.num_nodes, self.num_nodes)). \
                reshape(intra_graph.shape)
        graphs = torch.stack([inter_graph, intra_graph], dim=2)  # (T, B, 2, N, N)
        if gen_graph_only:
            # graphs = torch.threshold(graphs, 0.5, 0)
            return graphs, None
        if self.use_norm:
            reconst = self.sem_encoder(x_with_pre, spatial_norm_tensor(graphs, add_self_loops=False))
        else:
            reconst = self.sem_encoder(x_with_pre, graphs)
        return graphs, reconst


