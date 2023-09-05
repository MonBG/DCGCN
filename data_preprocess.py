from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
from utils.path_utils import base_dir


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def delete_all_zero_samples(x, y):
    # mask all zero samples (either in input or output sequence)
    # x: (B, T, N, D)
    # y: (B, T, N, D)
    valid_id = []
    for i in range(x.shape[0]):
        if np.sum(y[i, :, :, 0]) > 0.1:
            valid_id.append(i)
    print(f"Delete {x.shape[0] - len(valid_id)} useless samples")
    return x[valid_id, ...], y[valid_id, ...]


def generate_train_val_test(args, index_list=None):
    if not isinstance(args, dict):
        args = args.__dict__
    node_ratio = args['node_ratio']
    sample_ratio = args['sample_ratio']
    df = pd.read_hdf(os.path.join(base_dir, args['traffic_df_filename']))
    df.index.freq = df.index.inferred_freq
    if index_list is not None:
        df = df.iloc[:, index_list]
    else:
        num_nodes = df.shape[1]
        df = df.iloc[:, :int(num_nodes * node_ratio)].copy()
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )
    # x, y = delete_all_zero_samples(x, y)
    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    sample_reserved_num = int(num_samples * sample_ratio)
    x, y = x[:sample_reserved_num], y[:sample_reserved_num]

    num_samples = sample_reserved_num
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    x_train, y_train = delete_all_zero_samples(x_train, y_train)
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_val, y_val = delete_all_zero_samples(x_val, y_val)
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        if not os.path.exists(os.path.join(base_dir, args['output_dir'])):
            os.makedirs(os.path.join(base_dir, args['output_dir']))
        np.savez_compressed(
            os.path.join(base_dir, args['output_dir'], "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)
    print("Finish!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node_ratio", type=float, default=1., help="control the number of nodes in the processed dataset"
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=1., help="control the number of samples in the processed dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/METR-LA/N20", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
