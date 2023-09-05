import os
import torch
import numpy as np
import igraph as ig
import networkx as nx
from sklearn import metrics
import matplotlib.pyplot as plt
import random

from utils.plot_lib import heatmap


## GCN-related

def spatial_norm_tensor(edge_weight, add_self_loops=True):
    """
    spatial-based normalization of adjacency matrix
    :param edge_weight:
    :return:
    """
    assert torch.isnan(edge_weight).sum() == 0
    num_nodes = edge_weight.shape[-1]
    if add_self_loops:
        edge_weight = edge_weight + torch.eye(num_nodes).to(edge_weight.device)

    d = edge_weight.sum(-1)   # (..., num_nodes), sum by nodes
    d_inv = torch.pow(d.float(), -1)
    d_inv[d_inv == float('inf')] = 0
    return d_inv.unsqueeze(-1) * edge_weight  # diag(d_inv) * edge_weight


def spectral_norm_tensor(edge_weight, add_self_loops=True):
    """
    spectral-based normalization of Laplacian matrix
    :param edge_weight: tensor (..., num_nodes, num_nodes)
    :param num_nodes:
    :param add_self_loops: default to true
    :return:
    """
    if torch.isnan(edge_weight).sum() > 0:
        raise Exception('Found nan in gcn_norm input')
    num_nodes = edge_weight.shape[-1]
    if add_self_loops:
        edge_weight = edge_weight + torch.eye(num_nodes).to(edge_weight.device)

    d = edge_weight.sum(-1).abs()  # (..., num_nodes)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
    d_inv_sqrt_kron = torch.einsum('...i, ...j->...ij', d_inv_sqrt, d_inv_sqrt)
    normalized_edge_weight = d_inv_sqrt_kron * edge_weight  # d_hat_inv_sqrt.matmul(adj).matmul(d_hat_inv_sqrt)
    return normalized_edge_weight.float()


def spectral_norm_numpy(edge_weight, add_self_loops=True):
    """
    :param edge_weight: ndarray (num_nodes, num_nodes)
    :param num_nodes:
    :param add_self_loops: default to true
    :return:
    """
    num_nodes = edge_weight.shape[0]
    if add_self_loops:
        edge_weight = edge_weight + np.eye(num_nodes)

    d = np.abs(edge_weight.sum(-1)) + 1e-5  # sum by column for each row
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_edge_weight = d_mat_inv_sqrt.dot(edge_weight.dot(d_mat_inv_sqrt))
    return normalized_edge_weight


## DAG reshape
def split_dag(W, d, lag=1):
    """ From (L*N, L*N) to (lag+1, L-lag, N, N)
    :param W:
    :param d:
    :param lag:
    :return:
    """

    def get_W_ti_tj(i, j):
        # i, j: 0,...,L-1
        return W[d * i:d * (i + 1), d * j:d * (j + 1)]

    # W: (L*d, L*d)
    # n: number of nodes
    L = int(W.shape[0] / d)
    split_graphs = {'intra': []}
    for k in range(lag):
        split_graphs[f'inter{k + 1}'] = []

    for t in range(lag, L):
        for k in range(lag):
            split_graphs[f'inter{k + 1}'].append(get_W_ti_tj(t - (k + 1), t))
        split_graphs['intra'].append(get_W_ti_tj(t, t))
    for key in split_graphs.keys():
        split_graphs[key] = np.stack(split_graphs[key], 0)  # (L-lag, d, d)
    return split_graphs


def merge_dag(W_split, lag=1):
    """ Reverse function of split_dag
    """
    intras = W_split['intra']
    L = intras.shape[0] + lag
    d = intras.shape[1]
    W_true = np.zeros((L * d, L * d))
    for t in range(lag, L):
        for k in range(lag):
            W_true[(t - (k + 1)) * d:(t - k) * d, t * d:(t + 1) * d] = W_split[f'inter{k + 1}'][t - lag, ...]
        W_true[t * d:(t + 1) * d, t * d:(t + 1) * d] = W_split['intra'][t - lag, ...]
    return W_true


### Metrics
def random_dag(num_nodes, prob):
    """ Generate random DAG
    :param num_nodes:
    :param prob:
    :return: np.ndarray (num_nodes, num_nodes)
    """
    G = nx.gnp_random_graph(num_nodes, prob, directed=True)
    DAG = nx.DiGraph()
    DAG.add_nodes_from(range(num_nodes))
    DAG.add_edges_from([(u, v, {'weight': random.random()}) for (u, v) in G.edges() if u < v])
    return nx.to_numpy_array(DAG)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def cal_graph_accuracy(B_true, B_est, N=5):
    """ Compute various accuracy metrics for DAG
    :param B_true (np.ndarray): [d, d] or [L, N, N], ground truth graph, {0, 1}
    :param B_est (np.ndarray): [d, d] or [L, N, N], estimated graph, {0, 1}
    :param N (int): number of nodes
    :return:
    """
    if isinstance(B_true, torch.Tensor):
        B_true = B_true.detach().cpu().numpy()
    if isinstance(B_est, torch.Tensor):
        B_est = B_est.detach().cpu().numpy()
    if len(B_true.shape) == 3:
        # (L, N, 2N): (inter_graph, intra_graph)
        N = B_true.shape[1]
        split_graphs = {
            'inter1': B_true[..., :N],
            'intra': B_true[..., N:2 * N]
        }
        B_true = merge_dag(split_graphs, lag=1)
    if len(B_est.shape) == 3:
        # (L, N, 2N): (inter_graph, intra_graph)
        N = B_est.shape[1]
        split_graphs = {
            'inter1': B_est[..., :N],
            'intra': B_est[..., N:2 * N]
        }
        B_est = merge_dag(split_graphs, lag=1)
    d = B_true.shape[0]
    # print(f"Number of nodes is {d}")
    mask_pos = np.flatnonzero(merge_dag({'inter1': np.ones((N, N)), 'intra': np.ones((N, N))}))  # pos of

    if not is_dag(B_est):
        print("Warning: B_est is not a DAG!!!")

    # linear index of non-zeros (by row)
    pred = np.flatnonzero(B_est)  # pos of directed edges for B_est / predicted positive
    cond = np.flatnonzero(B_true)  # pos of directed edges for B_true / true positive
    cond_reversed = np.flatnonzero(B_true.T)  # transpose to consider skeleton i-j and j-i are same in skeleton
    # cond_skeleton = np.concatenate([cond, cond_reversed])  # concat by row

    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)  # true positive

    # false pos
    false_pos = np.setdiff1d(pred, cond, assume_unique=True)  # false positive

    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)  # : return the pos in pred that are not in cond
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)  # right in skeleton but with reversed direction

    # compute ratio
    pred_size = len(pred)
    # cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    L = int((d / N) - 1)
    cond_neg_size = L * (0.5 * N * (N - 1) + N ** 2) - len(cond)  # maximum number of edges - number of real edges
    accuracy = metrics.accuracy_score(B_true.reshape(-1)[mask_pos], B_est.reshape(-1)[mask_pos])
    precision = metrics.precision_score(B_true.reshape(-1)[mask_pos], B_est.reshape(-1)[mask_pos], average='micro')
    tpr = float(len(true_pos)) / max(len(cond), 1)  # true positive rate
    fdr = float(len(false_pos)) / max(pred_size, 1)  # false discovery rate: FP/(FP+TP)
    fpr = float(len(false_pos)) / max(cond_neg_size, 1)  # false positive rate: (FP / (FP+TN))

    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    return {
        'acc': accuracy, 'ppv': precision,
        'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size
    }


## Graph generator Evaluation
def evaluate_graph_generator(A_inters: torch.Tensor, A_intras: torch.Tensor,
                             ground_truth: torch.Tensor, threshold=0.01, logger=None):
    ground_truth_binary = ground_truth.abs() > threshold
    mean_acc_list = {
        'acc': [],
        'ppv': [],
        'fdr': [],
        'tpr': [],
        'fpr': [],
        'shd': [],
        'nnz': []
    }
    for i in range(A_inters.shape[1]):
        pred_graphs = torch.cat([A_inters[:, i, ...], A_intras[:, i, ...]], -1)
        pred_graphs_binary = pred_graphs.abs() > threshold
        ret = cal_graph_accuracy(ground_truth_binary, pred_graphs_binary)
        for key in mean_acc_list.keys():
            mean_acc_list[key].append(ret[key])
    message = "acc {:.3f}, ppv {:.3f}, fdr {:.3f}, tpr {:.3f}, fpr {:.3f}, shd {:.0f}, nnz {:.0f}".format(
        np.mean(mean_acc_list['acc']), np.mean(mean_acc_list['ppv']),
        np.mean(mean_acc_list['fdr']), np.mean(mean_acc_list['tpr']), np.mean(mean_acc_list['fpr']),
        np.mean(mean_acc_list['shd']), np.mean(mean_acc_list['nnz'])
    )
    for key in mean_acc_list.keys():
        print(f'{key} std: {np.std(mean_acc_list[key])}')
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def plot_graph_generator(A_inters: torch.Tensor, A_intras: torch.Tensor, plot_dir):
    # A_inters, A_intras: (T, B, N, N)
    for t in range(A_inters.shape[0]):
        A_inter_t, A_intra_t = A_inters[t, 0, ...], A_intras[t, 0, ...]

        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        inter_graph = A_inter_t.abs().cpu().numpy()
        intra_graph = A_intra_t.abs().cpu().numpy()
        heatmap(inter_graph, cmap='Blues', ax=ax[0],
                title=f't{t}: DBN-inter', cbar_kw={'fraction': 0.05})
        heatmap(intra_graph, cmap='Blues', ax=ax[1],
                title=f't{t}: DBN-intra', cbar_kw={'fraction': 0.05})
        plt.savefig(os.path.join(plot_dir, f'graph_slice_t{t}.png'))


def plot_multi_slice_graphs(graphs: torch.Tensor, plot_dir):
    # graphs: (T, m, N, N)
    num_nodes = graphs.shape[-1]
    ratio = num_nodes // 20
    graphs = graphs.cpu().abs()
    seq_len = graphs.shape[0]
    num_graph = graphs.shape[1]
    for t in range(seq_len):
        fig, ax = plt.subplots(1, num_graph, figsize=(8 * ratio * num_graph, 9 * ratio))
        for i in range(num_graph):
            graph = graphs[t, i, ...].numpy()
            heatmap(graph, cmap='Blues', ax=ax[i],
                    title=f't{t}: slice {num_graph - i - 1}', cbar_kw={'fraction': 0.05})
        plt.savefig(os.path.join(plot_dir, f'graph_slice_t{t}.png'))


def plot_multi_slice_multi_gen_graphs(graphs: torch.Tensor, plot_dir, num_generator=2):
    # graphs: (T, G, m, N, N)
    assert num_generator > 1
    assert graphs.shape[1] == num_generator
    graphs = graphs.cpu().abs()
    seq_len = graphs.shape[0]
    num_graph = graphs.shape[2]
    for t in range(seq_len):
        fig, ax = plt.subplots(num_generator, num_graph, figsize=(8 * num_graph, 9 * num_generator))
        for i in range(num_generator):
            for j in range(num_graph):
                graph = graphs[t, i, j, ...].numpy()
                heatmap(graph, cmap='Blues', ax=ax[i, j],
                        title=f't{t}: Head {i}, slice {num_graph - j - 1}', cbar_kw={'fraction': 0.05})
            plt.savefig(os.path.join(plot_dir, f'graph_slice_t{t}.png'))
