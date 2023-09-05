import torch
import torch.nn as nn


class DGBlockV1(nn.Module):
    """
    Simple MLP
    ---
    Modification:
    - 02210950:
        Delete BatchNormX before activation;
        Change activation from gelu to relu;
        decrease the # of layers by 1
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DGBlockV1, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (T, B, N, D) or (N, B, D)
        return self.fc(x)


class DAGGumbelSigmoid(nn.Module):
    def __init__(self, tau=0.2):
        super(DAGGumbelSigmoid, self).__init__()
        self.tau = tau
        self.test_num = 6

    @staticmethod
    def init_sample_gumbel(shape, device="cuda", eps=1e-20):
        u = torch.rand(shape).to(device)
        u = -torch.log(-torch.log(u + eps) + eps)
        # u[np.arange(shape[0]), np.arange(shape[0])] = 0  # set diagonal to zero
        return u

    def forward(self, x, mask=False):
        # x: (T, B, N, N)
        if self.training:
            diff_gumbel = self.init_sample_gumbel(x.shape, device=x.device) - \
                          self.init_sample_gumbel(x.shape, device=x.device)
            output = gumbel_sigmoid(x, diff_gumbel, temperature=self.tau)
        else:
            x_re = x.repeat(self.test_num, 1, 1, 1, 1)
            diff_gumbel = self.init_sample_gumbel(x_re.shape, device=x_re.device) - \
                          self.init_sample_gumbel(x_re.shape, device=x_re.device)
            output = gumbel_sigmoid(x_re, diff_gumbel, temperature=self.tau).mean(dim=0)

        if mask:
            diagonal_mask = (1. - torch.eye(x.shape[-1])).to(x.device)  # (N, N)
            output = diagonal_mask * output

        return output


def gumbel_sigmoid(logits, diff_gumbel, temperature):
    gumbel_softmax_sample = logits + diff_gumbel
    y = torch.sigmoid(gumbel_softmax_sample / temperature)
    return y