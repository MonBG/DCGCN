import torch
import numpy as np
from utils.metrics_lib import MAE_torch, RMSE_torch, MSE_torch, masked_mae, masked_rmse, masked_mse


def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if labels.max() < (mask_value + 1e-1):
            mae = masked_mae(preds, labels, mask_value)
        else:
            mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss


def masked_rmse_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if labels.max() < (mask_value + 1e-1):
            mae = masked_rmse(preds, labels, mask_value)
        else:
            mae = RMSE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss


def masked_mse_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if labels.max() < (mask_value + 1e-1):
            mae = masked_mse(preds, labels, mask_value)
        else:
            mae = MSE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss


# def masked_mae_loss(y_pred, y_true):
#     mask = (y_true != 0).float()
#     mask /= mask.mean()
#     loss = torch.abs(y_pred - y_true)
#     loss = loss * mask
#     # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
#     loss[loss != loss] = 0
#     return loss.mean()


def graph_sparsity_loss(graphs):
    """
    :param graphs: (seq_len, batch_size, m, num_nodes, num_nodes)
    :return:
    """
    loss = torch.abs(graphs).sum((-1, -2)).mean()
    return loss


def graph_segment_change_loss(graphs):
    # graphs: (T, B, 2, N, N)
    loss = torch.abs(graphs[1:, ...] - graphs[:-1, ...]).sum((-1, -2)).mean()
    return loss


def masked_causal_sem_loss(x_with_pre, A_inters, A_intras, mask=None):
    """
    :param x: (seq_len + 1, batch_size, num_nodes, in_feats_dim)
    :param A_inters: (seq_len, batch_size, num_nodes, num_nodes)
    :param A_intras: (seq_len, batch_size, num_nodes, num_nodes)
    :return:
    """
    pre_x = x_with_pre[:-1, ...]
    x = x_with_pre[1:, ...]

    # multiplication hatX = XA, each column of A should be normalized
    xhat = torch.einsum('tbik,tbij->tbjk', pre_x, A_inters) + torch.einsum('tbik,tbij->tbjk', x, A_intras)
    loss = torch.square(x - xhat)
    if mask is not None:
        mask /= mask.mean()
        loss = loss * mask
    loss[loss != loss] = 0
    return loss.sum((2, 3)).mean()


def acyclic_loss(A_intras, approx=True):
    """
    :param A_intras:
    :return: (seq_len, batch_size, num_nodes, in_feats_dim)
    """
    def matrix_poly(matrix, d):
        x = torch.eye(d).to(matrix.device) + torch.div(matrix, d)
        return torch.matrix_power(x, d)

    seq_len, batch_size, num_nodes, _ = A_intras.size()
    loss = None
    for t in range(seq_len):
        for b in range(batch_size):
            if approx:
                expm_A = matrix_poly(A_intras[t, b, ...] * A_intras[t, b, ...], num_nodes)
                cur_loss = torch.trace(expm_A) - num_nodes
            else:
                cur_loss = torch.matrix_exp(torch.square(A_intras[t, b, ...])).trace() - num_nodes

            if loss is None:
                loss = cur_loss
            else:
                loss += cur_loss
    return loss / (seq_len * batch_size)


def kl_div_gauss(mu0, log_var0, mu1, log_var1, reduction='sum'):
    # for vae: 0 means Q(z|x), 1 means prior
    kl_div_element = log_var1 - log_var0 + (log_var0.exp() + (mu1 - mu0).pow(2))/log_var1.exp() - 1
    if reduction == 'sum':
        return 0.5 * torch.sum(kl_div_element)
    elif reduction == 'mean':
        return 0.5 * torch.mean(kl_div_element)


def nll_gauss(x, mu, log_var, mask=None, reduction='sum'):
    """ log-likelihood of x given gaussian distribution parameters mu and log_var
    """
    if x.shape != mu.shape:
        raise Exception("nll_gauss: x.shape != mu.shape")
    # nll_elememnt = 0.5 * log_var + 0.5 * np.log(np.pi) + (x - mu).pow(2) / (2 * log_var.exp())
    nll_elememnt = 0.5 * log_var + (x - mu).pow(2) / (2 * log_var.exp())
    if mask is not None:
        mask /= mask.mean()
        nll_elememnt = nll_elememnt * mask
    if reduction == 'sum':
        return torch.sum(nll_elememnt)
    elif reduction == 'mean':
        return torch.mean(nll_elememnt)


def mse_loss(x, mu, mask=None, reduction='sum'):
    """square loss of gaussian distribution, equal to nll_loss if log_var is equal to identity
    """
    if x.shape != mu.shape:
        raise Exception("mse_loss: x.shape != mu.shape")
    se_element = (x - mu).pow(2)
    if mask is not None:
        mask /= mask.mean()
        se_element = se_element * mask
    if reduction == 'sum':
        return torch.sum(se_element)
    elif reduction == 'mean':
        return torch.mean(se_element)
