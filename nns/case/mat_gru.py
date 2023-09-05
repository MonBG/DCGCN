import torch
import torch.nn as nn
import math


class MatGRU(nn.Module):
    """
    Multi-layer Graph generator (positive coefficients only)

    Notes:
        1. only the first layer considers the pre_input
    """

    def __init__(self, row_dim, in_feats_dim, hidden_dim, num_layers=2):
        super(MatGRU, self).__init__()
        self.row_dim = row_dim
        self.in_feats_dim = in_feats_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.mat_gru_cells = nn.ModuleList()
        self.mat_gru_cells.append(MatGRUCell(row_dim, in_feats_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.mat_gru_cells.append(MatGRUCell(row_dim, hidden_dim, hidden_dim))

    def init_hidden(self, batch_size):
        init_state_list = []
        for i in range(self.num_layers):
            init_state = self.mat_gru_cells[i].init_hidden(batch_size)
            init_state_list.append(init_state)
        init_states = torch.stack(init_state_list, dim=0)
        return init_states  # (num_layers, B, N, H)

    def forward(self, x, init_states=None):
        # inputs_with_pre: (T, B, N, D), N=row_dim
        # init_states: (nlayer, B, N, H)
        current_inputs = x
        seq_len, batch_size, _, _ = current_inputs.size()  # T+1

        if init_states is None:
            init_states = self.init_hidden(batch_size).to(current_inputs.device)

        output_hidden = []
        for i in range(self.num_layers):
            state = init_states[i]
            inner_states = []
            for t in range(seq_len):
                state = self.mat_gru_cells[i](current_inputs[t, ...], state)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=0)  # (T, B, N, H)

        output_hidden = torch.stack(output_hidden, dim=0)  # (num_layers, B, N, H)
        return current_inputs, output_hidden


class MatGRUCell(nn.Module):
    """
    Graph generator gru cell for single time-slice
    """

    def __init__(self, row_dim, in_feats_dim, hidden_dim):
        super(MatGRUCell, self).__init__()
        # dim of hidden state is set to num_nodes
        self.row_dim = row_dim
        self.in_feats_dim = in_feats_dim
        self.hidden_dim = hidden_dim

        self.update = MatGRUGate(row_dim, in_feats_dim, hidden_dim, nn.Sigmoid())
        self.reset = MatGRUGate(row_dim, in_feats_dim, hidden_dim, nn.Sigmoid())
        self.htilda = MatGRUGate(row_dim, in_feats_dim, hidden_dim, nn.Tanh())
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.row_dim, self.hidden_dim)

    def forward(self, X, hidden):
        # X: (B, N, D)
        # hidden: (B, N, H)
        update = self.update(X, hidden)
        reset = self.reset(X, hidden)

        h_cap = reset * hidden
        h_cap = self.htilda(X, h_cap)

        new_hidden = (1 - update) * hidden + update * h_cap

        # layer norm
        new_hidden = self.layer_norm(new_hidden)

        return new_hidden


class MatGRUGate(nn.Module):
    def __init__(self, row_dim, in_feats_dim, hidden_dim, activation):
        super(MatGRUGate, self).__init__()
        self.row_dim = row_dim
        self.in_feats_dim = in_feats_dim
        self.hidden_dim = hidden_dim

        self.activation = activation
        self.W = nn.Parameter(torch.FloatTensor(in_feats_dim, hidden_dim))
        self.U = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.FloatTensor(row_dim, hidden_dim))
        self.reset_parameters()

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def reset_parameters(self):
        # nn.init.xavier_normal_(self.W)
        # nn.init.xavier_normal_(self.U)
        self.reset_param(self.W)
        self.reset_param(self.U)
        nn.init.constant_(self.bias, val=0.)

    def forward(self, X, hidden):
        """
        :param X: shape (batch_size, num_nodes, in_feats_dim)
        :param hidden: shape (batch_size, num_nodes, hidden_dim)
        :return:
        """
        out = X.matmul(self.W) + hidden.matmul(self.U) + self.bias
        out = self.activation(out)
        return out