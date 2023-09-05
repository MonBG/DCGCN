import torch.nn as nn
from nns.case.utils_blocks import DyGCN


class MaskGCNSEM(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, hidden_dim):
        super(MaskGCNSEM, self).__init__()
        self.num_nodes = num_nodes
        self.num_layers = 4
        self.pre_fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True))
        self.block_inter = nn.ModuleList([DyGCN(num_nodes, hidden_dim, hidden_dim, hidden_dim)
                                          for _ in range(self.num_layers)])
        self.block_intra = nn.ModuleList([DyGCN(num_nodes, hidden_dim, hidden_dim, hidden_dim)
                                          for _ in range(self.num_layers)])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_with_pre, graphs):
        # x_with_pre: (T+1, B, N, D)
        # graphs: (T, B, 2, N, N), almost binary mask
        x_with_pre = self.pre_fc(x_with_pre)  # (T+1, B, N, H)

        pre_x, x = x_with_pre[:-1, ...], x_with_pre[1:, ...]  # (T, B, N ,H)
        inter_graph, intra_graph = graphs.permute(2, 0, 1, 3, 4)  # (T, B, N, N)
        T, B, _, _ = inter_graph.shape

        for i in range(self.num_layers):
            pre_x = self.block_inter[i](pre_x, inter_graph) + pre_x  # skip connection, (T, B, N, H)
        for i in range(self.num_layers):
            x = self.block_inter[i](x, inter_graph) + x  # skip connection, (T, B, N, H)

        out = self.fc_out(pre_x + x).reshape(T, B, self.num_nodes, -1)  # (T, B, N, F)
        return out
