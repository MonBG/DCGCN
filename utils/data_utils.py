"""
Generate utils for data loading and processing
"""

import os
import pickle

import torch
import numpy as np

from utils.path_utils import base_dir


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def data_loader(X, Y, batch_size, shuffle=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def load_data(data_dir, batch_size, sample_ratio=1.):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        if sample_ratio < 1.:
            sample_size = cat_data['x'].shape[0]
            if category == 'train':
                id = np.random.permutation(range(sample_size))[:int(sample_size * sample_ratio)]
            else:
                id = np.arange(int(sample_size * sample_ratio))
            data['x_' + category] = data['x_' + category][id, ...]
            data['y_' + category] = data['y_' + category][id, ...]

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    print('Train', data['x_train'].shape, data['y_train'].shape)
    print('Val', data['x_val'].shape, data['y_val'].shape)
    print('Test', data['x_test'].shape, data['y_test'].shape)
    data['train_loader'] = data_loader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = data_loader(data['x_val'], data['y_val'], batch_size, shuffle=False)
    data['test_loader'] = data_loader(data['x_test'], data['y_test'], batch_size, shuffle=False)
    data['scaler'] = scaler
    return data


def align_data_dbgcn2dcrnn(inputs, targets, seq_len=12, horizon=12):
    # inputs: tensor [B, N, T=1, D_in]
    # targets: tensor [B, N, T=1, D_out]
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    inputs_aligned = []
    targets_aligned = []
    # moving window: seq_len + horizon, note that inputs.shape[0] = L-1
    num_samples = inputs.shape[0] + 1 - (seq_len + horizon) + 1
    for i in range(num_samples):
        inputs_aligned.append(inputs[i:(i+seq_len), ...].permute(2, 0, 1, 3))  # each (1, seq_len, N, D)
        targets_aligned.append(targets[(i+seq_len-1):(i+seq_len-1+horizon), ...].permute(2, 0, 1, 3))
    inputs_aligned = torch.cat(inputs_aligned, 0)
    targets_aligned = torch.cat(targets_aligned, 0)
    print("input: ", inputs_aligned.shape, "target:", targets_aligned.shape)
    # inputs_aligned: tensor [B, seq_len, N, D_in]
    # targets_aligned: tensor [B, horizon, N, D_out]
    return inputs_aligned, targets_aligned


def align_data_dbgcn2tvdbgcn(inputs, targets, seq_len=12, lag=1):
    # inputs: tensor [B, N, T=1, D_in]
    # targets: tensor [B, N, T=1, D_out]
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    inputs_aligned = []
    targets_aligned = []
    num_samples = inputs.shape[0] - (seq_len + lag) + 1  # moving window: seq_len + 1
    for i in range(num_samples):
        inputs_aligned.append(inputs[i:(i+seq_len+lag), ...].permute(2, 0, 1, 3))  # each (1, seq_len, N, D)
        targets_aligned.append(targets[(i+lag):(i+seq_len+lag), ...].permute(2, 0, 1, 3))
    inputs_aligned = torch.cat(inputs_aligned, 0)
    targets_aligned = torch.cat(targets_aligned, 0)
    print("input: ", inputs_aligned.shape, "target:", targets_aligned.shape)
    # inputs_aligned: tensor [B, seq_len+lag, N, D_in]
    # targets_aligned: tensor [B, seq_len, N, D_out]
    return inputs_aligned, targets_aligned


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_graph(graph_pkl_filename, num_nodes=None):
    sensor_ids, sensor_id_to_ind, adj_mat = load_pickle(graph_pkl_filename)
    if num_nodes is not None:
        assert adj_mat.shape[0] >= num_nodes
        adj_mat = adj_mat[:num_nodes, :num_nodes]
    return adj_mat


if __name__ == "__main__":
    os.chdir('D:\Code_Files\Pycharm\Traffic')
    data_dir = os.path.join(base_dir, 'data/METR-LA-DBGCN/N207')
    data_dir_dcrnn = os.path.join(base_dir, 'data/METR-LA/N207')
    data_dir_tvdbgcn = os.path.join(base_dir, 'data/METR-LA-TVDBGCN/N207')
    gen_dcrnn, gen_tvdbgcn = True, False
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    for cat in ["train", "val", "test"]:
        if gen_dcrnn:
            print("Generating dcrnn data")
            if not os.path.exists(data_dir_dcrnn):
                os.makedirs(data_dir_dcrnn)
            _x, _y = align_data_dbgcn2dcrnn(data["x_" + cat], data["y_" + cat])
            np.savez_compressed(
                os.path.join(data_dir_dcrnn, "%s.npz" % cat),
                x=_x.numpy(),
                y=_y.numpy(),
            )

        if gen_tvdbgcn:
            print("Generating tbdbgcn data")
            if not os.path.exists(data_dir_tvdbgcn):
                os.makedirs(data_dir_tvdbgcn)
            _x, _y = align_data_dbgcn2tvdbgcn(data["x_" + cat], data["y_" + cat], lag=3)
            np.savez_compressed(
                os.path.join(data_dir_tvdbgcn, "%s.npz" % cat),
                x=_x.numpy(),
                y=_y.numpy(),
            )

    print("Finish!")