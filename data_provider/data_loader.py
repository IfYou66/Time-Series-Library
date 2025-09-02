import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    用于处理以下数据集的Dataset类：
        时间序列分类档案库 (www.timeseriesclassification.com)
    参数：
        limit_size: float类型，范围(0, 1)，用于调试时限制数据集大小
    属性：
        all_df: (num_samples * seq_len, num_columns) 数据框，按整数索引索引，多个行对应同一个索引（样本）。
            每一行是一个时间步；每一列包含元数据（例如时间戳）或特征。
        feature_df: (num_samples * seq_len, feat_dim) 数据框；包含`all_df`中对应选定特征的列子集
        feature_names: `feature_df`中包含的列名（与feature_df.columns相同）
        all_IDs: (num_samples,) 包含在`all_df`/`feature_df`中的ID序列（与all_df.index.unique()相同）
        labels_df: (num_samples, num_labels) 每个样本标签的pandas DataFrame
        max_seq_len: 最大序列（时间序列）长度。如果为None，将使用脚本参数`max_seq_len`。
            （此外，脚本参数会覆盖此属性）
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        """
        初始化UEA数据加载器
        关键步骤：
        1. 加载数据：从.ts文件加载原始数据
        2. 样本ID提取：获取所有唯一的样本ID
        3. 数据集限制：支持调试时使用部分数据
        4. 特征设置：使用所有列作为特征
        5. 标准化：对数据进行标准化处理
        """
        self.args = args
        self.root_path = root_path
        self.flag = flag
        
        # 1. 加载数据：从.ts文件加载原始数据
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        
        # 2. 获取样本ID：所有样本ID（整数索引 0 ... num_samples-1）
        self.all_IDs = self.all_df.index.unique()

        # 3. 限制数据集大小（用于调试）
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # 如果在(0, 1]范围内，解释为比例
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # 4. 使用所有特征
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # 5. 数据预处理：标准化
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        从`root_path`中包含的ts文件加载数据集到数据框中，可选择从`pattern`中选择
        参数：
            root_path: 包含所有单独.ts文件的目录
            file_list: 可选地，提供`root_path`内要考虑的文件路径列表。
                否则，将使用整个`root_path`内容。
        返回：
            all_df: 包含指定文件所有数据的单个（可能连接的）数据框
            labels_df: 包含每个样本标签的数据框
        """
        # 选择训练和评估的路径
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # 所有路径的列表
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('未找到文件: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("使用模式'{}'未找到.ts文件".format(pattern))

        # 单个文件包含整个数据集
        all_df, labels_df = self.load_single(input_paths[0])

        return all_df, labels_df

    def load_single(self, filepath):
        """
        加载单个.ts文件并处理数据
        关键处理步骤：
        1. 标签处理：将类别标签转换为数字编码
        2. 长度检查：处理变长序列问题
        3. 数据重构：将数据转换为长格式
        4. 缺失值处理：使用插值填充缺失值
        """
        # 1. 从.ts文件加载数据
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        
        # 2. 处理标签：将类别标签转换为数字编码
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories  # 获取类别名称
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32在使用nn.CrossEntropyLoss时会产生错误

        # 3. 检查序列长度：(num_samples, num_dimensions) 包含每个序列长度的数组
        lengths = df.applymap(
            lambda x: len(x)).values

        # 4. 处理变长序列：检查同一样本的不同维度长度是否不同
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # 如果任何行（样本）在不同维度上有不同的长度
            df = df.applymap(subsample)  # 进行子采样

        # 5. 确定最大序列长度
        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # 如果任何列（维度）在不同样本中有不同的长度
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # 6. 数据重构：首先为每个样本创建一个(seq_len, feat_dim)数据框，按单个整数索引（样本的"ID"）
        # 然后连接成(num_samples * seq_len, feat_dim)数据框，多行对应样本索引
        # （即与此项目中所有数据集相同的方案）

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # 7. 处理缺失值：替换NaN值
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        """
        实例标准化：对特定数据集进行特殊的实例标准化
        特点：对EthanolConcentration数据集进行特殊的数值稳定性处理
        """
        if self.root_path.count('EthanolConcentration') > 0:  # 为数值稳定性进行特殊处理
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        """
        获取单个样本的数据
        关键特点：
        1. 按索引获取：通过样本ID获取对应的特征和标签
        2. 数据增强：支持训练时的数据增强
        3. 格式转换：将numpy数组转换为torch张量
        """
        # 1. 获取特征和标签
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        
        # 2. 数据增强（仅在训练时且增强比例大于0）
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            
            # 重塑为3D格式进行增强
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        # 3. 返回标准化后的数据：将numpy数组转换为torch张量
        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        """
        返回数据集中的样本数量
        """
        return len(self.all_IDs)


class Dataset_Hell(Dataset):
    """
    Hell Bridge Test Arena 多分类数据集加载器
    每个 CSV 对应一个状态（10 类），
    将每条长时序按固定窗口 seq_len 划分，并在 TRAIN/VAL/TEST 三种模式下完成 70%/15%/15% 划分。
    """

    def __init__(self, args, root_path, flag='TRAIN'):
        super().__init__()
        self.args = args
        self.root = root_path
        self.flag = flag.upper()
        assert self.flag in ('TRAIN','VAL','TEST'), f"flag must be TRAIN/VAL/TEST, got {flag}"

        # 1) 找到所有 CSV
        pattern = os.path.join(self.root, 'MVS_P2_*.csv')
        self.file_paths = sorted(glob.glob(pattern))
        assert len(self.file_paths)==10, f"Expected 10 CSVs, found {len(self.file_paths)}"

        # 2) 生成每个文件的标签
        self.labels = []
        self.class_names = []
        for p in self.file_paths:
            fn = os.path.basename(p)
            if 'UDS_NM_Z_01' in fn:    lbl=0
            elif 'UDS_NM_Z_02' in fn:  lbl=1
            else:
                m=re.search(r'DS([1-8])',fn)
                assert m, f"Bad filename {fn}"
                lbl=int(m.group(1))+1
            self.labels.append(lbl)
            self.class_names.append(fn)

        # 3) 读首个文件拿总长度 & 特征维
        df0 = pd.read_csv(self.file_paths[0],header=None)
        total_len, feat_dim = df0.shape

        # 4) 计算非重叠窗口数
        self.seq_len = args.seq_len
        n = total_len // self.seq_len
        assert n>0, "seq_len太大，得不到任何窗口"

        # 5) 按比例划分：round 避免全部向下取整
        train_r, val_r, test_r = 0.7, 0.15, 0.15
        n_train = max(1, round(n*train_r))
        n_val   = max(1, round(n*val_r))
        # 剩余给 test
        n_test  = n - n_train - n_val
        # 若出现负数（四舍五入导致），再调整
        if n_test<1:
            n_test=1
            # 重新保证总和
            if n_train + n_val + n_test > n:
                # 优先保证 train，再 val，再 test
                overflow = n_train+n_val+n_test - n
                for _ in range(overflow):
                    if n_test>1: n_test-=1
                    elif n_val>1: n_val-=1
                    else: n_train = max(1,n_train-1)

        # 6) 读入所有数据
        self.data = []
        for p in self.file_paths:
            df = pd.read_csv(p,header=None)
            self.data.append(df.values)
        self.data = np.stack(self.data,axis=0)  # (10, total_len, feat_dim)
        self.windows_per_file = n

        # 7) 构造本 flag 下的全局 idx 列表
        self.indices = []
        for f_idx in range(len(self.file_paths)):
            if self.flag=='TRAIN':
                wins = range(0, n_train)
            elif self.flag=='VAL':
                wins = range(n_train, n_train+n_val)
            else:  # TEST
                wins = range(n_train+n_val, n)
            for w in wins:
                self.indices.append(f_idx*n + w)

        # 8) 供 Exp_Classification 读取用
        self.max_seq_len = self.seq_len
        self.feature_df  = pd.DataFrame(np.zeros((1, feat_dim)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        g = self.indices[idx]
        f_idx = g // self.windows_per_file
        w_idx = g %  self.windows_per_file
        arr   = self.data[f_idx]  # (total_len,feat_dim)
        start = w_idx * self.seq_len
        seg   = arr[start:start+self.seq_len]
        x = torch.from_numpy(seg).float()
        y = torch.tensor(self.labels[f_idx],dtype=torch.long)
        return x, y

class Dataset_Van(Dataset):
    """
    三分类数据集加载器（type00/01/02）
    - 每个 CSV 对应一个类别（0/1/2）
    - 将长序列按固定窗口 seq_len 非重叠切分
    - TRAIN / VAL / TEST = 70% / 15% / 15%
    - 返回: (x, y) 其中 x: [seq_len, feat_dim]，y: 标量类别ID (0/1/2)
    """

    def __init__(self, args, root_path, flag='TRAIN'):
        super().__init__()
        self.args = args
        self.root = root_path
        self.flag = str(flag).upper()
        assert self.flag in ('TRAIN', 'VAL', 'TEST'), f"flag must be TRAIN/VAL/TEST, got {flag}"

        # 1) 匹配三个 CSV 文件
        pattern = os.path.join(self.root, 'type*.csv')
        self.file_paths = sorted(glob.glob(pattern))
        assert len(self.file_paths) == 3, f"Expected 3 CSVs, found {len(self.file_paths)} at {pattern}"

        # 2) 生成标签：type00->0, type01->1, type02->2
        self.labels = []
        self.class_names = []
        for p in self.file_paths:
            fn = os.path.basename(p).lower()
            m = re.search(r'type(\d{2})', fn)
            assert m, f"Bad filename (expect type00/01/02*): {fn}"
            lbl = int(m.group(1))
            assert lbl in (0, 1, 2), f"Label must be 00/01/02, got {lbl:02d} from {fn}"
            self.labels.append(lbl)
            self.class_names.append(fn)

        # 3) 读取所有文件为 float32，并对齐长度与列数（鲁棒）
        arrays = []
        min_len, min_feat = None, None
        for p in self.file_paths:
            arr = self._read_csv_numeric_array(p)  # float32, 无缺失, C 连续
            arrays.append(arr)
            L, C = arr.shape
            min_len = L if min_len is None else min(min_len, L)
            min_feat = C if min_feat is None else min(min_feat, C)

        # 对齐到共同最短长度与最小列数，避免越界/不一致
        arrays = [arr[:min_len, :min_feat] for arr in arrays]
        self.data = np.stack(arrays, axis=0)  # (3, total_len, feat_dim) float32
        total_len, feat_dim = min_len, min_feat

        # 4) 计算非重叠窗口数
        self.seq_len = int(self.args.seq_len)
        assert self.seq_len > 0, "seq_len must be positive"
        n = total_len // self.seq_len
        assert n > 0, f"seq_len({self.seq_len}) 太大，单文件无法切出任何窗口（total_len={total_len}）"

        # 5) 按窗口数划分 TRAIN/VAL/TEST 比例（四舍五入并确保和为 n）
        train_r, val_r, test_r = 0.7, 0.15, 0.15
        n_train = max(1, round(n * train_r))
        n_val   = max(1, round(n * val_r))
        n_test  = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_train + n_val + n_test > n:
                overflow = n_train + n_val + n_test - n
                for _ in range(overflow):
                    if n_test > 1:
                        n_test -= 1
                    elif n_val > 1:
                        n_val -= 1
                    else:
                        n_train = max(1, n_train - 1)

        self.windows_per_file = n

        # 6) 根据 flag 生成全局索引
        self.indices = []
        for f_idx in range(len(self.file_paths)):
            if self.flag == 'TRAIN':
                wins = range(0, n_train)
            elif self.flag == 'VAL':
                wins = range(n_train, n_train + n_val)
            else:  # TEST
                wins = range(n_train + n_val, n)
            for w in wins:
                self.indices.append(f_idx * n + w)

        # 7) 供 Exp_Classification 占位属性（与框架保持一致）
        self.max_seq_len = self.seq_len
        self.feature_df  = pd.DataFrame(np.zeros((1, feat_dim), dtype=np.float32))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        g = self.indices[idx]
        f_idx = g // self.windows_per_file   # 文件索引 (0/1/2)
        w_idx = g %  self.windows_per_file   # 窗口索引
        arr   = self.data[f_idx]             # (total_len, feat_dim) float32
        start = w_idx * self.seq_len
        seg   = arr[start:start + self.seq_len]          # (seq_len, feat_dim) float32
        seg   = np.ascontiguousarray(seg, dtype=np.float32)  # C 连续，避免 from_numpy 额外拷贝
        x = torch.from_numpy(seg)                         # torch.float32
        y = torch.tensor(self.labels[f_idx], dtype=torch.long)
        return x, y

    @staticmethod
    def _read_csv_numeric_array(path):
        """
        鲁棒读取 CSV 为 float32 的 numpy 数组：
        - 以 header=0 读取（首行为列名）
        - 丢掉时间戳列(ts)，保留数值列
        - to_numeric 强制数值化，非法值置 NaN
        - 缺失填充：ffill -> bfill -> 0
        - 返回 C 连续 float32
        """
        try:
            df = pd.read_csv(path, header=0, engine='c', dtype=str)
        except Exception:
            df = pd.read_csv(path, header=0, engine='python', dtype=str)

        # 删除时间戳列（一般叫 'ts'）
        if 'ts' in df.columns:
            df = df.drop(columns=['ts'])

        # 去掉全空列 & 去空白
        df = df.dropna(axis=1, how='all').applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # 强制数值化
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # 缺失值处理
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)

        # 转为 float32 & C 连续
        arr = df.to_numpy(dtype=np.float32, copy=False)
        return np.ascontiguousarray(arr, dtype=np.float32)
