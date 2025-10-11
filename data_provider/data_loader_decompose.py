import os
import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features
import numpy as np

warnings.filterwarnings("ignore")


class Dataset_ETT_hour_Decompose(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        history_len=24 * 4 * 4,
        overlap_len=24 * 4,
        pred_len=24 * 4,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        period=24,
    ):
        self.history_len = history_len
        self.overlap_len = overlap_len
        self.pred_len = pred_len

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.period = period
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = (
            0,
            12 * 30 * 24 - self.history_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.history_len,
        )
        border2s = (
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        )
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.history_len
        r_begin = s_end - self.overlap_len
        r_end = r_begin + self.overlap_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        x_trend, x_seasonal = self.decompose(seq_x)
        y_trend, y_seasonal = self.decompose(seq_y)

        return (
            seq_x,
            seq_y,
            x_trend,
            x_seasonal,
            y_trend,
            y_seasonal,
            seq_x_mark,
            seq_y_mark,
        )

    def __len__(self):
        return len(self.data_x) - self.history_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def decompose(self, data):
        """
        Decompose the time series data into trend, seasonal, and residual components.
        """
        kernel_size = self.period + 1
        pad = (kernel_size - 1) // 2
        kernel = np.ones(kernel_size) / kernel_size
        trend_est = np.apply_along_axis(
            lambda seq: np.convolve(
                np.pad(seq, pad, mode="reflect"), kernel, mode="valid"
            ),
            axis=0,
            arr=data,
        )
        seasonal_est = data - trend_est

        return trend_est, seasonal_est


class Dataset_ETT_hour_Seasonal(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        history_len=24 * 4 * 4,
        overlap_len=24 * 4,
        pred_len=24 * 4,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        period=24,
    ):
        self.history_len = history_len
        self.overlap_len = overlap_len
        self.pred_len = pred_len

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.period = period
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = (
            0,
            12 * 30 * 24 - self.history_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.history_len,
        )
        border2s = (
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        )
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.history_len
        r_begin = s_end - self.overlap_len
        r_end = r_begin + self.overlap_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        x_trend, x_seasonal = self.decompose(seq_x)
        y_trend, y_seasonal = self.decompose(seq_y)

        return (
            x_seasonal,
            y_seasonal,
            seq_x_mark,
            seq_y_mark,
        )

    def __len__(self):
        return len(self.data_x) - self.history_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def decompose(self, data):
        kernel_size = self.period + 1
        pad = (kernel_size - 1) // 2
        kernel = np.ones(kernel_size) / kernel_size
        trend_est = np.apply_along_axis(
            lambda seq: np.convolve(
                np.pad(seq, pad, mode="reflect"), kernel, mode="valid"
            ),
            axis=0,
            arr=data,
        )
        seasonal_est = data - trend_est

        return trend_est, seasonal_est


class Dataset_Custom_Decompose(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        history_len=24 * 4 * 4,
        overlap_len=24 * 4,
        pred_len=24 * 4,
        features="S",
        data_path="custom.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        period=24,
    ):
        self.history_len = history_len
        self.overlap_len = overlap_len
        self.pred_len = pred_len

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.period = period
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0,
            num_train - self.history_len,
            len(df_raw) - num_test - self.history_len,
        ]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """Get item

        Parameters
        ----------
        index : int
            Index of the item to get

        Returns
        -------
        seq_x : torch.Tensor
            Sequence of past values
        seq_y : torch.Tensor
            Sequence of future values
        seq_x_mark : torch.Tensor
            Sequence of past timestamps
        seq_y_mark : torch.Tensor
            Sequence of future timestamps
        """
        s_begin = index
        s_end = s_begin + self.history_len
        r_begin = s_end - self.overlap_len
        r_end = r_begin + self.overlap_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        x_trend, x_seasonal = self.decompose(seq_x)
        y_trend, y_seasonal = self.decompose(seq_y)

        return (
            seq_x,
            seq_y,
            x_trend,
            x_seasonal,
            y_trend,
            y_seasonal,
            seq_x_mark,
            seq_y_mark,
        )

    def __len__(self):
        return len(self.data_x) - self.history_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def decompose(self, data):
        kernel_size = self.period + 1
        pad = (kernel_size - 1) // 2
        kernel = np.ones(kernel_size) / kernel_size
        trend_est = np.apply_along_axis(
            lambda seq: np.convolve(
                np.pad(seq, pad, mode="reflect"), kernel, mode="valid"
            ),
            axis=0,
            arr=data,
        )
        seasonal_est = data - trend_est

        return trend_est, seasonal_est


class Dataset_Custom_Seasonal(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        history_len=24 * 4 * 4,
        overlap_len=24 * 4,
        pred_len=24 * 4,
        features="S",
        data_path="custom.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        period=24,
    ):
        self.history_len = history_len
        self.overlap_len = overlap_len
        self.pred_len = pred_len

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.period = period
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0,
            num_train - self.history_len,
            len(df_raw) - num_test - self.history_len,
        ]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """Get item

        Parameters
        ----------
        index : int
            Index of the item to get

        Returns
        -------
        seq_x : torch.Tensor
            Sequence of past values
        seq_y : torch.Tensor
            Sequence of future values
        seq_x_mark : torch.Tensor
            Sequence of past timestamps
        seq_y_mark : torch.Tensor
            Sequence of future timestamps
        """
        s_begin = index
        s_end = s_begin + self.history_len
        r_begin = s_end - self.overlap_len
        r_end = r_begin + self.overlap_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        x_trend, x_seasonal = self.decompose(seq_x)
        y_trend, y_seasonal = self.decompose(seq_y)

        return (
            x_seasonal,
            y_seasonal,
            seq_x_mark,
            seq_y_mark,
        )

    def __len__(self):
        return len(self.data_x) - self.history_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def decompose(self, data):
        kernel_size = self.period + 1
        pad = (kernel_size - 1) // 2
        kernel = np.ones(kernel_size) / kernel_size
        trend_est = np.apply_along_axis(
            lambda seq: np.convolve(
                np.pad(seq, pad, mode="reflect"), kernel, mode="valid"
            ),
            axis=0,
            arr=data,
        )
        seasonal_est = data - trend_est

        return trend_est, seasonal_est
