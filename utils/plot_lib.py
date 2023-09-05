import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import math


def heatmap(data, row_labels=None, col_labels=None,
            ax=None, title=None,
            cbar_kw={}, cbarlabel="",
            show_grid=True, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if col_labels is not None:
        ax.set_xticklabels(col_labels)
    if row_labels is not None:
        ax.set_yticklabels(row_labels)
    if title is not None:
        ax.set_title(title)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    if show_grid:
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def display_seq_pred(outputs, target, plot_dim=0, plot_range=None, mfrow=None, figsize=(24, 15),
                     title=None, save_path=None, figname='output'):
    """
    :param outputs: tensor or ndarray (seq_len, num_nodes=20, dim)
    :param target: tensor or ndarray (seq_len, num_nodes=20, dim)
    :return:
    """
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    if plot_range is not None:
        begin, end = plot_range
        outputs = outputs[begin:end, ...]
        target = target[begin:end, ...]

    if target.shape[1] != outputs.shape[1]:
        raise Exception("The number of nodes for target and outputs do not match!")
    num_nodes = target.shape[1]
    if mfrow is None:
        nrow = math.floor(math.sqrt(num_nodes))
        ncol = math.ceil(num_nodes / nrow)
    else:
        nrow, ncol = mfrow

    seq_len, _, _ = outputs.shape
    t_index = np.arange(0, seq_len).astype(np.int64)

    fig, ax = plt.subplots(nrow, ncol, figsize=figsize, sharex='all')
    for i in range(num_nodes):
        row_id, col_id = divmod(i, ncol)
        ax[row_id, col_id].plot(t_index, target[:, i, plot_dim], label='truth')
        ax[row_id, col_id].plot(t_index, outputs[:, i, plot_dim], label='pred')
        ax[row_id, col_id].set_title(f'N{i}')
        ax[row_id, col_id].legend()
    fig.text(0.5, 0.08, 'Time', ha='center', fontsize=figsize[0] * .8)
    fig.text(0.09, 0.5, 'Speed', va='center', rotation='vertical', fontsize=figsize[0] * .8)
    if title is not None:
        fig.suptitle(title, fontsize=figsize[0] * .8)
    fig.tight_layout()
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{figname}.png'))
    else:
        plt.show()


def display_dbn_graph(A_inter, A_intra, title=None,
                      record=True, figname='output_graph', save_path='data/figs'):
    """
    :param A_inter: tensor (num_nodes, num_nodes)
    :param A_intra: tensor (num_nodes, num_nodes)
    :return:
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    inter_graph = A_inter.abs().numpy()
    intra_graph = A_intra.abs().numpy()
    max_inter = inter_graph.max()
    max_intra = intra_graph.max()
    heatmap(inter_graph / max_inter, cmap='Blues', ax=ax[0],
            title='DBN-inter', vmin=0, vmax=1, cbar_kw={'fraction': 0.05})
    heatmap(intra_graph / max_intra, cmap='Blues', ax=ax[1],
            title='DBN-intra', vmin=0, vmax=1, cbar_kw={'fraction': 0.05})
    if title is not None:
        fig.suptitle(f't {1}: {title}', fontsize=10)
    if record:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f'{save_path}/{figname}.png')
    else:
        plt.show()


def align_seq2seq_output(output, dataset='dbgcn', lag=1):
    # align to first step of dcrnn output
    # output: Tensor (seq_len, batch_size, num_nodes, dim)
    # output_aligned: Tensor (seq_len, num_nodes, dim)
    if dataset == 'dbgcn':
        L = output.shape[1] + 1
        dcrnn_size = L - 24 + 1
        begin_index = 11
        end_index = begin_index + dcrnn_size
        return output[0, begin_index:end_index, ...]  # first step prediction
    elif dataset == 'tvdbgcn':
        # first step of tvdbgcn output (Note! GRU type should be the last step)
        L = output.shape[1] + 12 + lag
        dcrnn_size = L - 24 + 1

        # first step output
        begin_index = 11 - lag
        end_index = begin_index + dcrnn_size
        return output[0, begin_index:end_index, ...]  # first step prediction

