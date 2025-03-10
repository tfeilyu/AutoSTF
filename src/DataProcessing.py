import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler

from src.utils.helper import Scaler
from src import train_util

data_path = './dataset'


class TFdataProcessing:
    def __init__(self, dataset, train_prop, valid_prop,
                 num_sensors, in_length, out_length, in_channels, batch_size_per_gpu):

        self.traffic_data = {}
        self.dataset = dataset
        self.num_sensors = num_sensors

        self.train_prop = train_prop
        self.valid_prop = valid_prop

        self.in_length = in_length
        self.out_length = out_length
        self.in_channels = in_channels
        self.batch_size = batch_size_per_gpu
        self.adj_type = 'doubletransition'

        self.nodeID = self.read_idx()

        # get several adjacency matrix
        self.adj_mx_01, self.adj_mx_dcrnn, self.adj_mx_gwn = self.read_adj_mat()

        self.dataloader = {}

        # self.build_graph()
        self.build_data_loader()

    def build_graph(self):
        logging.info('initialize graph')

        for dim in range(self.adj_mats.shape[-1]):
            values = self.adj_mats[:, :, dim][self.adj_mats[:, :, dim] != np.inf].flatten()
            self.adj_mats[:, :, dim] = np.exp(-np.square(self.adj_mats[:, :, dim] / (values.std() + 1e-8)))

        # normalize node_ft
        # self.node_fts = self.read_loc()
        # self.node_fts = (self.node_fts - self.node_fts.mean(axis=0)) / (self.node_fts.std(axis=0) + 1e-8)

    def build_data_loader(self):
        logging.info('initialize data loader')

        train_traffic, valid_traffic, test_traffic = self.read_traffic()
        train_data = train_traffic[list(self.nodeID.keys())]
        self.scaler = Scaler(train_data.values, missing_value=0)

        # data for search
        self.search_train = self.get_data_loader(train_traffic, shuffle=True, tag='search_train')  # for weight update
        self.search_valid = self.get_data_loader(valid_traffic, shuffle=True, tag='search_valid')  # for arch update

        # data for training & evaluation
        self.train = self.get_data_loader(train_traffic, shuffle=True, tag='train')
        self.valid = self.get_data_loader(valid_traffic, shuffle=False, tag='valid')
        self.test = self.get_data_loader(test_traffic, shuffle=False, tag='test')

    def get_data_loader(self, data, shuffle, tag):
        if len(data) == 0:
            return 0

        num_timestamps = data.shape[0]

        data_time = data.iloc[:, 0]
        self.traffic_data[tag+'_data'] = data

        data = data[list(self.nodeID.keys())]

        # fill missing value
        data_fill = self.fill_traffic(data)

        # transform data distribution
        data_f = np.expand_dims(self.scaler.transform(data_fill.values), axis=-1)  # [T, N, 1]

        # time in day
        time_ft = (pd.to_datetime(data_time.values) - data_time.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_ft = np.tile(time_ft, [1, self.num_sensors, 1]).transpose((2, 1, 0))  # [T, N, 1]

        # day in week
        day_ft = pd.to_datetime(data_time.values).dayofweek # [T, N, 1]
        day_ft = np.tile(day_ft, [1, self.num_sensors, 1]).transpose((2, 1, 0))

        # put all input features together
        if self.in_channels == 1:
            in_data = data_f
        elif self.in_channels == 2:
            in_data = np.concatenate([data_f, time_ft], axis=-1)  # [T, N, D]
        elif self.in_channels == 3:
            in_data = np.concatenate([data_f, time_ft, day_ft], axis=-1)  # [T, N, D]
        else:
            print('channels error')
            sys.exit()

        out_data = np.expand_dims(data.values, axis=-1)  # [T, N, 1]

        # create inputs & labels
        inputs, labels = [], []
        for i in range(self.in_length):
            temp = in_data[i: num_timestamps + 1 - self.in_length - self.out_length + i]
            inputs += [temp]
        for i in range(self.out_length):
            temp = out_data[self.in_length + i: num_timestamps + 1 - self.out_length + i]
            labels += [temp]
        # inputs = np.stack(inputs).transpose((1, 3, 2, 0))
        # labels = np.stack(labels).transpose((1, 3, 2, 0))
        inputs = np.stack(inputs).transpose((1, 0, 2, 3))
        labels = np.stack(labels).transpose((1, 0, 2, 3))

        # logging info of inputs & labels
        logging.info('load %s inputs & labels [ok]', tag)
        logging.info('input shape: %s', inputs.shape)  # [num_timestamps, c, n, input_len]
        logging.info('label shape: %s', labels.shape)  # [num_timestamps, c, n, output_len]

        # create dataset
        dataset = TensorDataset(
            torch.from_numpy(inputs).to(dtype=torch.float),
            torch.from_numpy(labels).to(dtype=torch.float)
        )

        # create sampler
        sampler = SequentialSampler(dataset)
        if shuffle:
            sampler = RandomSampler(dataset, replacement=True, num_samples=self.batch_size)
        else:
            sampler = SequentialSampler(dataset)

        # create dataloader
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=sampler,
                                 num_workers=4, drop_last=False)

        self.dataloader[tag+'_loader'] = DataLoaderM(inputs, labels, self.batch_size)
        self.dataloader['x_'+tag] = inputs
        self.dataloader['y_'+tag] = labels

        return data_loader

    def read_idx(self):
        with open(os.path.join(data_path, self.dataset, self.dataset+'_sensor_id.txt')) as f:
            ids = f.read().strip().split('\n')
        idx = {}
        for i, id in enumerate(ids):
            idx[id] = i
        return idx

    def read_adj_mat(self):
        # 更改 邻接矩阵只保留距离信息

        graph_csv = pd.read_csv(os.path.join(data_path, self.dataset, self.dataset+'_distances.csv'),
                                dtype={'from': 'str', 'to': 'str'})
        adj_distance_dcrnn = np.zeros((self.num_sensors, self.num_sensors))
        adj_distance_dcrnn[:] = np.inf  # 无穷大

        # 0, 1 adjacency matrix
        adj_mx_01 = np.zeros((self.num_sensors, self.num_sensors))
        for k in range(self.num_sensors):
            adj_mx_01[k, k] = 1

        for row in graph_csv.values:
            if row[0] in self.nodeID and row[1] in self.nodeID:
                adj_distance_dcrnn[self.nodeID[row[0]], self.nodeID[row[1]]] = row[2]  # distance
                adj_mx_01[self.nodeID[row[0]], self.nodeID[row[1]]] = 1  # 0, 1

        distances = adj_distance_dcrnn[~np.isinf(adj_distance_dcrnn)].flatten()
        std = distances.std()
        adj_mx = np.exp(-np.square(adj_distance_dcrnn / std))
        # Make the adjacent matrix symmetric by taking the max.
        # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

        # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
        adj_mx[adj_mx < 0.1] = 0
        adj_mx_dcrnn = adj_mx

        # GraphWaveNet matrix
        adj_mx_gwn = [train_util.asym_adj(adj_mx), train_util.asym_adj(np.transpose(adj_mx))]


        return adj_mx_01, adj_mx_dcrnn, adj_mx_gwn

    def read_traffic(self):
        # data = pd.read_hdf(os.path.join(data_path, self._path, 'traffic.h5'))
        data = pd.read_csv(os.path.join(data_path, self.dataset, self.dataset+'_data.csv'))
        self.data_time = data.iloc[:, 0]

        num_train = int(data.shape[0] * self.train_prop)
        num_valid = int(data.shape[0] * self.valid_prop)
        num_test = data.shape[0] - num_train - num_valid

        train = data[:num_train].copy()
        valid = data[num_train: num_train + num_valid].copy()
        test = data[-num_test:].copy()

        return train, valid, test

    def fill_traffic(self, data):
        # data = data[list(self.nodeID.keys())]
        data = data.copy()
        data[data < 1e-5] = float('nan')
        data = data.fillna(method='pad')
        data = data.fillna(method='bfill')
        return data


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()