import os
from abc import abstractmethod
from typing import Union, Tuple

import h5py
import numpy as np
import scipy.io as scio
import scipy.sparse
from scipy.sparse import load_npz
from sklearn.datasets import load_svmlight_file

from .benchmark_utils import Dataset

__all__ = (
    "get_load_fn", "get_cv_fn",
    "BaseDataset", "HiggsDataset", "SusyDataset", "MillionSongsDataset",
    "TimitDataset", "NycTaxiDataset", "YelpDataset", "FlightsDataset"
)


def load_from_npz(dset_name, folder, verbose=False):
    x_file = os.path.join(folder, "%s_data.npz" % dset_name)
    y_file = os.path.join(folder, "%s_target.npy" % dset_name)
    x_data = np.asarray(load_npz(x_file).todense())
    y_data = np.load(y_file)
    if verbose:
        print("Loaded %s. X: %s - Y: %s" % (dset_name, x_data.shape, y_data.shape))
    return (x_data, y_data)


def load_from_t(dset_name, folder, verbose=False):
    file_tr = os.path.join(folder, dset_name)
    file_ts = os.path.join(folder, dset_name + ".t")
    x_data_tr, y_data_tr = load_svmlight_file(file_tr)
    x_data_tr = np.asarray(x_data_tr.todense())
    x_data_ts, y_data_ts = load_svmlight_file(file_ts)
    x_data_ts = np.asarray(x_data_ts.todense())
    if verbose:
        print("Loaded %s. train X: %s - Y: %s - test X: %s - Y: %s" %
              (dset_name, x_data_tr.shape, y_data_tr.shape, x_data_ts.shape, y_data_ts.shape))
    x_data = np.concatenate((x_data_tr, x_data_ts))
    y_data = np.concatenate((y_data_tr, y_data_ts))
    return x_data, y_data


def standardize_x(Xtr, Xts):
    if isinstance(Xtr, np.ndarray):
        mXtr = Xtr.mean(axis=0, keepdims=True, dtype=np.float64).astype(Xtr.dtype)
        sXtr = Xtr.std(axis=0, keepdims=True, dtype=np.float64, ddof=1).astype(Xtr.dtype)
        sXtr[sXtr == 0] = 1.0
    else:
        mXtr = Xtr.mean(dim=0, keepdims=True)
        sXtr = Xtr.std(dim=0, keepdims=True)

    Xtr -= mXtr
    Xtr /= sXtr
    Xts -= mXtr
    Xts /= sXtr

    return Xtr, Xts, {}


def mean_remove_y(Ytr, Yts):
    mtr = np.mean(Ytr, dtype=np.float64).astype(Ytr.dtype)
    Ytr -= mtr
    Yts -= mtr
    Ytr = Ytr.reshape((-1, 1))
    Yts = Yts.reshape((-1, 1))
    return Ytr, Yts, {'Y_mean': mtr}


def standardize_y(Ytr, Yts):
    mtr = np.mean(Ytr, dtype=np.float64).astype(Ytr.dtype)
    stdtr = np.std(Ytr, dtype=np.float64, ddof=1).astype(Ytr.dtype)
    Ytr -= mtr
    Ytr /= stdtr
    Yts -= mtr
    Yts /= stdtr
    Ytr = Ytr.reshape((-1, 1))
    Yts = Yts.reshape((-1, 1))
    return Ytr, Yts, {'Y_mean': mtr, 'Y_std': stdtr}


def as_np_dtype(dtype):
    if "float32" in str(dtype):
        return np.float32
    if "float64" in str(dtype):
        return np.float64
    if "int32" in str(dtype):
        return np.int32
    raise ValueError(dtype)


def as_torch_dtype(dtype):
    import torch
    if "float32" in str(dtype):
        return torch.float32
    if "float64" in str(dtype):
        return torch.float64
    if "int32" in str(dtype):
        return torch.int32
    raise ValueError(dtype)


def equal_split(N, train_frac):
    Ntr = int(N * train_frac)
    idx = np.arange(N)
    np.random.shuffle(idx)
    idx_tr = idx[:Ntr]
    idx_ts = idx[Ntr:]
    return idx_tr, idx_ts


class MyKFold():
    def __init__(self, n_splits, shuffle, seed=92):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = np.random.RandomState(seed)

    def split(self, X, y=None):
        N = X.shape[0]
        indices = np.arange(N)
        mask = np.full(N, False)
        if self.shuffle:
            self.random_state.shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, N // n_splits, dtype=np.int)
        fold_sizes[:N % n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            mask.fill(False)
            mask[indices[start:stop]] = True
            yield mask
            current = stop


class BaseDataset():
    def load_data(self, dtype, as_torch=False, as_tf=False):
        X, Y = self.read_data(dtype)
        print(f"Loaded {self.dset_name()} dataset in {dtype} precision.", flush=True)
        Xtr, Ytr, Xts, Yts = self.split_data(X, Y, train_frac=None)
        assert Xtr.shape[0] == Ytr.shape[0]
        assert Xts.shape[0] == Yts.shape[0]
        assert Xtr.shape[1] == Xts.shape[1]
        print(f"Split the data into {Xtr.shape[0]} training, "
              f"{Xts.shape[0]} validation points of dimension {Xtr.shape[1]}.", flush=True)
        Xtr, Xts, other_X = self.preprocess_x(Xtr, Xts)
        Ytr, Yts, other_Y = self.preprocess_y(Ytr, Yts)
        print("Data-preprocessing completed.", flush=True)
        kwargs = dict()
        kwargs.update(other_X)
        kwargs.update(other_Y)
        if as_torch:
            return self.to_torch(Xtr, Ytr, Xts, Yts, **kwargs)
        if as_tf:
            return self.to_tensorflow(Xtr, Ytr, Xts, Yts, **kwargs)
        return Xtr, Ytr, Xts, Yts, kwargs

    def load_data_cv(self, dtype, k, as_torch=False):
        X, Y = self.read_data(dtype)
        print(f"Loaded {self.dset_name()} dataset in {dtype} precision.", flush=True)
        print(f"Data size: {X.shape[0]} points with {X.shape[1]} features", flush=True)

        kfold = MyKFold(n_splits=k, shuffle=True)
        iteration = 0
        for test_idx in kfold.split(X):
            Xtr = X[~test_idx]
            Ytr = Y[~test_idx]
            Xts = X[test_idx]
            Yts = Y[test_idx]
            Xtr, Xts, other_X = self.preprocess_x(Xtr, Xts)
            Ytr, Yts, other_Y = self.preprocess_y(Ytr, Yts)
            print("Preprocessing complete (iter %d) - Divided into %d train, %d test points" %
                  (iteration, Xtr.shape[0], Xts.shape[0]))
            kwargs = dict()
            kwargs.update(other_X)
            kwargs.update(other_Y)
            if as_torch:
                yield self.to_torch(Xtr, Ytr, Xts, Yts, **kwargs)
            else:
                yield Xtr, Ytr, Xts, Yts, kwargs
            iteration += 1

    @staticmethod
    @abstractmethod
    def read_data(dtype):
        pass

    @staticmethod
    @abstractmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        pass

    @staticmethod
    @abstractmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    @staticmethod
    @abstractmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Ytr, Yts, {}

    @staticmethod
    def to_torch(Xtr, Ytr, Xts, Yts, **kwargs):
        import torch
        #torch_kwargs = {k: torch.from_numpy(v) for k, v in kwargs.items()}
        torch_kwargs = kwargs
        return (
            torch.from_numpy(Xtr),
            torch.from_numpy(Ytr),
            torch.from_numpy(Xts),
            torch.from_numpy(Yts),
            torch_kwargs
        )

    @staticmethod
    def to_tensorflow(Xtr, Ytr, Xts, Yts, **kwargs):
        # By default tensorflow is happy with numpy arrays
        return (Xtr, Ytr, Xts, Yts, kwargs)

    @abstractmethod
    def dset_name(self) -> str:
        return "UNKOWN"


class MillionSongsDataset(BaseDataset):
    file_name = '/data/DATASETS/MillionSongs/YearPredictionMSD.mat'
    _dset_name = 'MillionSongs'

    @staticmethod
    def read_data(dtype) -> Tuple[np.ndarray, np.ndarray]:
        f = scio.loadmat(MillionSongsDataset.file_name)
        X = f['X'][:, 1:].astype(as_np_dtype(dtype))
        Y = f['X'][:, 0].astype(as_np_dtype(dtype))
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac=None):
        if train_frac == 'auto' or train_frac is None:
            idx_tr = np.arange(463715)
            idx_ts = np.arange(463715, 463715 + 51630)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)

        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Ytr, dtype=np.float64).astype(Ytr.dtype)
        sttr = np.std(Ytr, dtype=np.float64, ddof=1).astype(Ytr.dtype)
        Ytr -= mtr
        Ytr /= sttr
        Yts -= mtr
        Yts /= sttr
        Ytr = Ytr.reshape((-1, 1))
        Yts = Yts.reshape((-1, 1))
        return Ytr, Yts, {'Y_std': sttr, 'Y_mean': mtr}

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def dset_name(self):
        return self._dset_name


class NycTaxiDataset(BaseDataset):
    file_name = '/data/DATASETS/NYCTAXI/NYCTAXI.h5'
    _dset_name = 'TAXI'
    _default_train_frac = 0.8

    @staticmethod
    def read_data(dtype):
        h5py_file = h5py.File(NycTaxiDataset.file_name, 'r')
        X = np.array(h5py_file['X'], dtype=as_np_dtype(dtype))  # N x 9
        Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype))  # N x 1

        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = NycTaxiDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Xtr, axis=0, dtype=np.float64, keepdims=True).astype(Xtr.dtype)
        vtr = np.std(Xtr, axis=0, dtype=np.float64, ddof=1, keepdims=True).astype(Xtr.dtype)

        Xtr -= mtr
        Xtr /= vtr
        Xts -= mtr
        Xts /= vtr

        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Ytr, dtype=np.float64).astype(Ytr.dtype)
        sttr = np.std(Ytr, dtype=np.float64, ddof=1).astype(Ytr.dtype)
        Ytr -= mtr
        Ytr /= sttr
        Yts -= mtr
        Yts /= sttr
        return Ytr, Yts, {'Y_std': sttr}

    def dset_name(self):
        return self._dset_name


class HiggsDataset(BaseDataset):
    file_name = '/data/DATASETS/HIGGS_UCI/Higgs.mat'
    _dset_name = 'HIGGS'
    _default_train_frac = 0.8

    @staticmethod
    def read_data(dtype):
        h5py_file = h5py.File(HiggsDataset.file_name, 'r')
        arr = np.array(h5py_file['X'], dtype=as_np_dtype(dtype)).T
        X = arr[:, 1:]
        Y = arr[:, 0]
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = HiggsDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Xtr, axis=0, dtype=np.float64, keepdims=True).astype(Xtr.dtype)
        vtr = np.var(Xtr, axis=0, dtype=np.float64, ddof=1, keepdims=True).astype(Xtr.dtype)

        Xtr -= mtr
        Xtr /= vtr
        Xts -= mtr
        Xts /= vtr

        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Convert labels from 0, 1 to -1, +1"""
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1)), {}

    def dset_name(self):
        return self._dset_name


class TimitDataset(BaseDataset):
    file_name = '/data/DATASETS/TIMIT/TIMIT.mat'
    _dset_name = 'TIMIT'

    @staticmethod
    def read_data(dtype):
        f = scio.loadmat(TimitDataset.file_name)
        dtype = as_np_dtype(dtype)
        Xtr = np.array(f['Xtr'], dtype=dtype)
        Xts = np.array(f['Xts'], dtype=dtype)
        Ytr = np.array(f['Ytr'], dtype=dtype).reshape((-1, ))
        Yts = np.array(f['Yts'], dtype=dtype).reshape((-1, ))
        X = np.concatenate((Xtr, Xts), axis=0)
        Y = np.concatenate((Ytr, Yts), axis=0)

        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            # Default split recovers the original Xtr, Xts split
            idx_tr = np.arange(1124823)
            idx_ts = np.arange(1124823, 1124823 + 57242)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)

        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        n_classes = 144
        damping = 1 / (n_classes - 1)
        eye = np.eye(n_classes, dtype=as_np_dtype(Ytr.dtype))
        A = eye - damping + eye * damping
        # Ytr
        Ytr = A[Ytr.astype(np.int32), :]
        # Yts
        Yts = (Yts - 1) * 3
        Yts = A[Yts.astype(np.int32), :]
        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name


class YelpDataset(BaseDataset):
    file_name = '/data/DATASETS/YELP_Ben/YELP_Ben_OnlyONES.mat'
    _dset_name = 'YELP'
    _default_train_frac = 0.8

    @staticmethod
    def read_data(dtype):
        dtype = as_np_dtype(dtype)
        f = h5py.File(YelpDataset.file_name, 'r')
        X = scipy.sparse.csc_matrix((
            np.array(f['X']['data'], dtype),
            f['X']['ir'][...], f['X']['jc'][...])).tocsr(copy=False)
        Y = np.array(f['Y'], dtype=dtype).reshape((-1, 1))
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = YelpDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        # scaler = sklearn.preprocessing.StandardScaler(copy=False, with_mean=False, with_std=True)
        # Xtr = scaler.fit_transform(Xtr)
        # Xts = scaler.transform(Xts)
        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Ytr, Yts, {}

    @staticmethod
    def to_torch(Xtr, Ytr, Xts, Yts, **kwargs):
        from falkon.sparse.sparse_tensor import SparseTensor
        import torch
        return (SparseTensor.from_scipy(Xtr),
                torch.from_numpy(Ytr),
                SparseTensor.from_scipy(Xts),
                torch.from_numpy(Yts), {})

    @staticmethod
    def to_tensorflow(Xtr, Ytr, Xts, Yts, **kwargs):
        import tensorflow as tf

        def scipy2tf(X):
            # Uses same representation as pytorch
            # https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor
            coo = X.tocoo()
            indices = np.array([coo.row, coo.col]).transpose()
            return tf.SparseTensor(indices, coo.data, coo.shape)
        return (scipy2tf(Xtr),
                Ytr,
                scipy2tf(Xts),
                Yts,
                {})

    def dset_name(self):
        return self._dset_name


class FlightsDataset(BaseDataset):
    file_name = '/data/DATASETS/FLIGHTS/flights.hdf5'
    _dset_name = 'FLIGHTS'
    _default_train_frac = 0.666

    @staticmethod
    def read_data(dtype):
        h5py_file = h5py.File(FlightsDataset.file_name, 'r')
        X = np.array(h5py_file['X'], dtype=as_np_dtype(dtype))
        Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype))
        # Preprocessing independent of train/test
        # As for https://github.com/jameshensman/VFF/blob/master/experiments/airline/airline_additive_figure.py
        # 1. Convert time of day from hhmm to minutes since midnight
        #  ArrTime is column 7, DepTime is column 6
        X[:, 7] = 60*np.floor(X[:, 7]/100) + np.mod(X[:, 7], 100)
        X[:, 6] = 60*np.floor(X[:, 6]/100) + np.mod(X[:, 6], 100)
        # 2. remove flights with silly negative delays (small negative delays are OK)
        pos_delay_idx = np.where(Y > -60)[0]
        X = X[pos_delay_idx, :]
        Y = Y[pos_delay_idx, :]
        # 3. remove outlying flights in term of length (col 'AirTime' at pos 5)
        short_flight_idx = np.where(X[:, 5] < 700)[0]
        X = X[short_flight_idx, :]
        Y = Y[short_flight_idx, :]

        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = FlightsDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Ytr, dtype=np.float64).astype(Ytr.dtype)
        sttr = np.std(Ytr, dtype=np.float64, ddof=1).astype(Ytr.dtype)
        Ytr -= mtr
        Ytr /= sttr
        Yts -= mtr
        Yts /= sttr
        Ytr = Ytr.reshape((-1, 1))
        Yts = Yts.reshape((-1, 1))
        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name


class FlightsClsDataset(BaseDataset):
    file_name = '/data/DATASETS/FLIGHTS/flights.hdf5'
    _dset_name = 'FLIGHTS-CLS'
    _default_train_num = 100_000

    @staticmethod
    def read_data(dtype):
        h5py_file = h5py.File(FlightsDataset.file_name, 'r')
        X = np.array(h5py_file['X'], dtype=as_np_dtype(dtype))
        Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype))
        # Preprocessing independent of train/test
        # As for https://github.com/jameshensman/VFF/blob/master/experiments/airline/airline_additive_figure.py
        # 1. Convert time of day from hhmm to minutes since midnight
        #  ArrTime is column 7, DepTime is column 6
        X[:, 7] = 60*np.floor(X[:, 7]/100) + np.mod(X[:, 7], 100)
        X[:, 6] = 60*np.floor(X[:, 6]/100) + np.mod(X[:, 6], 100)
        # Turn regression into classification by thresholding delay or not delay:
        Y = (Y <= 0).astype(X.dtype)

        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = (X.shape[0] - FlightsClsDataset._default_train_num) / X.shape[0]
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Convert labels from 0, 1 to -1, +1"""
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1)), {}

    def dset_name(self):
        return self._dset_name


class SusyDataset(BaseDataset):
    file_name = '/data/DATASETS/SUSY/Susy.mat'
    _dset_name = 'SUSY'
    _default_train_frac = 0.8

    @staticmethod
    def read_data(dtype):
        with h5py.File(SusyDataset.file_name, "r") as f:
            arr = np.asarray(f['X'], dtype=as_np_dtype(dtype)).T
            X = arr[:, 1:]
            Y = arr[:, 0].reshape(-1, 1)
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = SusyDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Convert labels from 0, 1 to -1, +1"""
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1)), {}

    def dset_name(self):
        return self._dset_name


class CIFAR10Dataset(BaseDataset):
    file_name = "/data/DATASETS/CIFAR10/cifar10.mat"
    ts_file_name = "/data/DATASETS/CIFAR10/cifar10.t.mat"
    _dset_name = "CIFAR10"

    @staticmethod
    def read_data(dtype):
        # Read Training data
        data = scio.loadmat(CIFAR10Dataset.file_name)
        Xtr = data["Z"].astype(as_np_dtype(dtype)) / 255
        Ytr = data["y"].astype(as_np_dtype(dtype)).reshape((-1, ))
        # Read Testing data
        data = scio.loadmat(CIFAR10Dataset.ts_file_name)
        Xts = data["Z"].astype(as_np_dtype(dtype)) / 255
        Yts = data["y"].astype(as_np_dtype(dtype)).reshape((-1, ))
        # Merge
        X = np.concatenate((Xtr, Xts), axis=0)
        Y = np.concatenate((Ytr, Yts), axis=0)
        # Convert to RGB
        R = X[:, :1024]
        G = X[:, 1024:2048]
        B = X[:, 2048:3072]
        X = 0.2126 * R + 0.7152 * G + 0.0722 * B
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac):
        if train_frac is None:
            idx_tr = np.arange(0, 50000)
            idx_ts = np.arange(50000, 60000)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        n_classes = 10
        eye = np.eye(n_classes, dtype=as_np_dtype(Ytr.dtype))
        A = eye
        Ytr = A[Ytr.astype(np.int32), :]
        Yts = A[Yts.astype(np.int32), :]
        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name


class SVHNDataset(BaseDataset):
    file_name = "/data/DATASETS/SVHN/SVHN.mat"
    ts_file_name = "/data/DATASETS/SVHN/SVHN.t.mat"
    _dset_name = "SVHN"

    @staticmethod
    def read_data(dtype):
        # Read Training data
        data = scio.loadmat(SVHNDataset.file_name)
        Xtr = data["Z"].astype(as_np_dtype(dtype)) / 255
        Ytr = data["y"].astype(as_np_dtype(dtype)).reshape((-1, ))
        # Read Testing data
        data = scio.loadmat(SVHNDataset.ts_file_name)
        Xts = data["Z"].astype(as_np_dtype(dtype)) / 255
        Yts = data["y"].astype(as_np_dtype(dtype)).reshape((-1, ))
        # Merge
        X = np.concatenate((Xtr, Xts), axis=0)
        Y = np.concatenate((Ytr, Yts), axis=0)
        # Convert to RGB
        R = X[:, :1024]
        G = X[:, 1024:2048]
        B = X[:, 2048:3072]
        X = 0.2126 * R + 0.7152 * G + 0.0722 * B
        # Y -- for some reason it's 1 indexed
        Y = Y - 1
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac):
        if train_frac is None:
            idx_tr = np.arange(0, 73257)
            idx_ts = np.arange(73257, 73257 + 26032)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        n_classes = 10
        eye = np.eye(n_classes, dtype=as_np_dtype(Ytr.dtype))
        A = eye
        Ytr = A[Ytr.astype(np.int32), :]
        Yts = A[Yts.astype(np.int32), :]
        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name


class MnistSmallDataset(BaseDataset):
    file_name = "/data/DATASETS/misc/mnist.hdf5"
    _dset_name = "MNIST"

    @staticmethod
    def read_data(dtype):
        with h5py.File(MnistSmallDataset.file_name, 'r') as h5py_file:
            X_train = np.array(h5py_file['X_train'], dtype=as_np_dtype(dtype))
            Y_train = np.array(h5py_file['Y_train'], dtype=as_np_dtype(dtype))
            X_test = np.array(h5py_file['X_test'], dtype=as_np_dtype(dtype))
            Y_test = np.array(h5py_file['Y_test'], dtype=as_np_dtype(dtype))
        X = np.concatenate([X_train, X_test], axis=0)
        Y = np.concatenate([Y_train, Y_test], axis=0)
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac):
        if train_frac is None:
            idx_tr = np.arange(0, 60000)
            idx_ts = np.arange(60000, 70000)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        Xtr /= 255.0
        Xts /= 255.0
        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        n_classes = 10
        eye = np.eye(n_classes, dtype=as_np_dtype(Ytr.dtype))
        Ytr = eye[Ytr.astype(np.int32), :]
        Yts = eye[Yts.astype(np.int32), :]
        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name


class MnistDataset(BaseDataset):
    file_name = '/data/DATASETS/MNIST/mnist8m_normalized.hdf5'
    _dset_name = 'MNIST8M'
    num_train = 6750000
    num_test = 10_000

    @staticmethod
    def read_data(dtype):
        with h5py.File(MnistDataset.file_name, "r") as f:
            Xtr = np.array(f["X_train"], dtype=as_np_dtype(dtype))
            Ytr = np.array(f["Y_train"], dtype=as_np_dtype(dtype))
            Xts = np.array(f["X_test"], dtype=as_np_dtype(dtype))
            Yts = np.array(f["Y_test"], dtype=as_np_dtype(dtype))
        return np.concatenate((Xtr, Xts), 0), np.concatenate((Ytr, Yts), 0)

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            idx_tr = np.arange(MnistDataset.num_train)
            idx_ts = np.arange(MnistDataset.num_train, MnistDataset.num_train + MnistDataset.num_test)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        n_classes = 10
        damping = 1 / (n_classes)
        eye = np.eye(n_classes, dtype=as_np_dtype(Ytr.dtype))
        A = eye - damping  # + eye * damping

        Ytr = A[Ytr.astype(np.int32), :]
        Yts = A[Yts.astype(np.int32), :]

        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name


class SmallHiggsDataset(BaseDataset):
    file_name = '/data/DATASETS/HIGGS_UCI/higgs_for_ho.hdf5'
    _dset_name = 'HIGGSHO'

    @staticmethod
    def read_centers(dtype):
        with h5py.File(SmallHiggsDataset.file_name, 'r') as h5py_file:
            centers = np.array(h5py_file['centers'], dtype=as_np_dtype(dtype))
        return centers

    @staticmethod
    def read_data(dtype):
        with h5py.File(SmallHiggsDataset.file_name, 'r') as h5py_file:
            X_train = np.array(h5py_file['X_train'], dtype=as_np_dtype(dtype))
            Y_train = np.array(h5py_file['Y_train'], dtype=as_np_dtype(dtype))
            X_test = np.array(h5py_file['X_test'], dtype=as_np_dtype(dtype))
            Y_test = np.array(h5py_file['Y_test'], dtype=as_np_dtype(dtype))
        X = np.concatenate([X_train, X_test], axis=0)
        Y = np.concatenate([Y_train, Y_test], axis=0)
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            idx_tr = np.arange(10000)
            idx_ts = np.arange(10000, 30000)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Xtr, axis=0, dtype=np.float64, keepdims=True).astype(Xtr.dtype)
        vtr = np.var(Xtr, axis=0, dtype=np.float64, ddof=1, keepdims=True).astype(Xtr.dtype)

        Xtr -= mtr
        Xtr /= vtr
        Xts -= mtr
        Xts /= vtr

        centers = SmallHiggsDataset.read_centers(Xtr.dtype)
        centers -= mtr
        centers /= vtr

        return Xtr, Xts, {'centers': centers}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Convert labels from 0, 1 to -1, +1"""
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1)), {}

    def dset_name(self):
        return self._dset_name


class IctusDataset(BaseDataset):
    file_name = '/data/DATASETS/ICTUS/run_all.mat'
    _dset_name = 'ICTUS'
    _default_train_frac = 0.8

    @staticmethod
    def read_data(dtype):
        data_dict = scio.loadmat(IctusDataset.file_name)
        X = np.asarray(data_dict['X'], dtype=as_np_dtype(dtype))
        Y = np.asarray(data_dict['Y'], dtype=as_np_dtype(dtype))
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = IctusDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Xtr, axis=0, dtype=np.float64, keepdims=True).astype(Xtr.dtype)
        vtr = (1.0 / np.std(Xtr, axis=0, dtype=np.float64, ddof=1, keepdims=True)).astype(Xtr.dtype)

        Xtr -= mtr
        Xtr *= vtr
        Xts -= mtr
        Xts *= vtr

        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Convert labels from 0, 1 to -1, +1"""
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1)), {}

    def dset_name(self):
        return self._dset_name


class SyntheticDataset(BaseDataset):
    file_name = '/data/DATASETS/Synthetic0.1Noise.mat'
    _dset_name = 'SYNTH01NOISE'
    _default_train_frac = 0.5

    @staticmethod
    def read_data(dtype):
        data_dict = scio.loadmat(SyntheticDataset.file_name)
        X = np.asarray(data_dict['X'], dtype=as_np_dtype(dtype))
        Y = np.asarray(data_dict['Y'], dtype=as_np_dtype(dtype))
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = SyntheticDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1)), {}

    def dset_name(self):
        return self._dset_name


class ChietDataset(BaseDataset):
    file_name = '/data/DATASETS/weather/CHIET.hdf5'
    _dset_name = 'CHIET'
    _num_train = 26227
    _num_test = 7832

    @staticmethod
    def read_data(dtype):
        with h5py.File(ChietDataset.file_name, 'r') as h5py_file:
            X_train = np.array(h5py_file['X_train'], dtype=as_np_dtype(dtype))
            Y_train = np.array(h5py_file['Y_train'], dtype=as_np_dtype(dtype))
            X_test = np.array(h5py_file['X_test'], dtype=as_np_dtype(dtype))
            Y_test = np.array(h5py_file['Y_test'], dtype=as_np_dtype(dtype))
        X = np.concatenate([X_train, X_test], axis=0)
        Y = np.concatenate([Y_train, Y_test], axis=0)
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            idx_tr = np.arange(ChietDataset._num_train)
            idx_ts = np.arange(ChietDataset._num_train, ChietDataset._num_train + ChietDataset._num_test)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return mean_remove_y(Ytr, Yts)

    def dset_name(self):
        return self._dset_name


class EnergyDataset(BaseDataset):
    file_name = '/data/DATASETS/energy.hdf5'
    _dset_name = 'ENERGY'
    _default_train_frac = 0.8

    @staticmethod
    def read_data(dtype):
        with h5py.File(EnergyDataset.file_name, 'r') as h5py_file:
            X = np.array(h5py_file['X'], dtype=as_np_dtype(dtype))
            Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype))
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = EnergyDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return mean_remove_y(Ytr, Yts)

    def dset_name(self):
        return self._dset_name


class BostonDataset(BaseDataset):
    file_name = '/data/DATASETS/boston.hdf5'
    _dset_name = 'BOSTON'
    _default_train_frac = 0.8

    @staticmethod
    def read_data(dtype):
        with h5py.File(BostonDataset.file_name, 'r') as h5py_file:
            X = np.array(h5py_file['X'], dtype=as_np_dtype(dtype))
            Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype))
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = BostonDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return mean_remove_y(Ytr, Yts)

    def dset_name(self):
        return self._dset_name


class ProteinDataset(BaseDataset):
    file_name = '/data/DATASETS/protein.hdf5'
    _dset_name = 'PROTEIN'
    _default_train_frac = 0.8

    @staticmethod
    def read_data(dtype):
        with h5py.File(ProteinDataset.file_name, 'r') as h5py_file:
            X = np.array(h5py_file['X'], dtype=as_np_dtype(dtype))
            Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype))
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = ProteinDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return mean_remove_y(Ytr, Yts)

    def dset_name(self):
        return self._dset_name


class Kin40kDataset(BaseDataset):
    file_name = '/data/DATASETS/kin40k.hdf5'
    _dset_name = 'KIN40K'
    _num_train = 10_000
    _num_test = 30_000

    @staticmethod
    def read_data(dtype):
        with h5py.File(Kin40kDataset.file_name, 'r') as h5py_file:
            X_train = np.array(h5py_file['X_train'], dtype=as_np_dtype(dtype))
            Y_train = np.array(h5py_file['Y_train'], dtype=as_np_dtype(dtype))
            X_test = np.array(h5py_file['X_test'], dtype=as_np_dtype(dtype))
            Y_test = np.array(h5py_file['Y_test'], dtype=as_np_dtype(dtype))
        X = np.concatenate([X_train, X_test], axis=0)
        Y = np.concatenate([Y_train, Y_test], axis=0)
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            idx_tr = np.arange(Kin40kDataset._num_train)
            idx_ts = np.arange(Kin40kDataset._num_train, Kin40kDataset._num_train + Kin40kDataset._num_test)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return mean_remove_y(Ytr, Yts)

    def dset_name(self):
        return self._dset_name


class CodRnaDataset(BaseDataset):
    folder = '/data/DATASETS/libsvm/binary'
    _dset_name = 'cod-rna'
    _num_train = 59_535
    _num_test = 271_617

    @staticmethod
    def read_data(dtype):
        x_data, y_data = load_from_t(CodRnaDataset._dset_name, CodRnaDataset.folder)
        x_data = x_data.astype(as_np_dtype(dtype))
        y_data = y_data.astype(as_np_dtype(dtype))
        return x_data, y_data

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            idx_tr = np.arange(CodRnaDataset._num_train)
            idx_ts = np.arange(CodRnaDataset._num_train, CodRnaDataset._num_train + CodRnaDataset._num_test)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        Ytr = Ytr.reshape(-1, 1)
        Yts = Yts.reshape(-1, 1)
        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name


class SvmGuide1Dataset(BaseDataset):
    folder = '/data/DATASETS/libsvm/binary'
    _dset_name = 'svmguide1'
    _num_train = 3089
    _num_test = 4000

    @staticmethod
    def read_data(dtype):
        x_data, y_data = load_from_t(SvmGuide1Dataset._dset_name, SvmGuide1Dataset.folder)
        x_data = x_data.astype(as_np_dtype(dtype))
        y_data = y_data.astype(as_np_dtype(dtype))
        return x_data, y_data

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            idx_tr = np.arange(SvmGuide1Dataset._num_train)
            idx_ts = np.arange(SvmGuide1Dataset._num_train, SvmGuide1Dataset._num_train + SvmGuide1Dataset._num_test)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Convert labels from 0, 1 to -1, +1"""
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1)), {}

    def dset_name(self):
        return self._dset_name


class PhishingDataset(BaseDataset):
    folder = '/data/DATASETS/libsvm/binary'
    _dset_name = 'phishing'
    _default_train_frac = 0.7

    @staticmethod
    def read_data(dtype):
        x_data, y_data = load_from_npz(PhishingDataset._dset_name, PhishingDataset.folder)
        x_data = x_data.astype(as_np_dtype(dtype))
        y_data = y_data.astype(as_np_dtype(dtype))
        return x_data, y_data

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = PhishingDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}  # No preproc, all values are equal-.-

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Convert labels from 0, 1 to -1, +1"""
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1)), {}

    def dset_name(self):
        return self._dset_name


class SpaceGaDataset(BaseDataset):
    folder = '/data/DATASETS/libsvm/regression'
    _dset_name = 'space_ga'
    _default_train_frac = 0.7

    @staticmethod
    def read_data(dtype):
        x_data, y_data = load_from_npz(SpaceGaDataset._dset_name, SpaceGaDataset.folder)
        x_data = x_data.astype(as_np_dtype(dtype))
        y_data = y_data.astype(as_np_dtype(dtype))
        return x_data, y_data

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = SpaceGaDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return mean_remove_y(Ytr, Yts)

    def dset_name(self):
        return self._dset_name


class CadataDataset(BaseDataset):
    folder = '/data/DATASETS/libsvm/regression'
    _dset_name = 'cadata'
    _default_train_frac = 0.7

    @staticmethod
    def read_data(dtype):
        x_data, y_data = load_from_npz(CadataDataset._dset_name, CadataDataset.folder)
        x_data = x_data.astype(as_np_dtype(dtype))
        y_data = y_data.astype(as_np_dtype(dtype))
        return x_data, y_data

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = CadataDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return mean_remove_y(Ytr, Yts)

    def dset_name(self):
        return self._dset_name


class MgDataset(BaseDataset):
    folder = '/data/DATASETS/libsvm/regression'
    _dset_name = 'mg'
    _default_train_frac = 0.7

    @staticmethod
    def read_data(dtype):
        x_data, y_data = load_from_npz(MgDataset._dset_name, MgDataset.folder)
        x_data = x_data.astype(as_np_dtype(dtype))
        y_data = y_data.astype(as_np_dtype(dtype))
        return x_data, y_data

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = MgDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return mean_remove_y(Ytr, Yts)

    def dset_name(self):
        return self._dset_name


class CpuSmallDataset(BaseDataset):
    folder = '/data/DATASETS/libsvm/regression'
    _dset_name = 'cpusmall'
    _default_train_frac = 0.7

    @staticmethod
    def read_data(dtype):
        x_data, y_data = load_from_npz(CpuSmallDataset._dset_name, CpuSmallDataset.folder)
        x_data = x_data.astype(as_np_dtype(dtype))
        y_data = y_data.astype(as_np_dtype(dtype))
        return x_data, y_data

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = CpuSmallDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return mean_remove_y(Ytr, Yts)

    def dset_name(self):
        return self._dset_name


class AbaloneDataset(BaseDataset):
    folder = '/data/DATASETS/libsvm/regression'
    _dset_name = 'abalone'
    _default_train_frac = 0.7

    @staticmethod
    def read_data(dtype):
        x_data, y_data = load_from_npz(AbaloneDataset._dset_name, AbaloneDataset.folder)
        x_data = x_data.astype(as_np_dtype(dtype))
        y_data = y_data.astype(as_np_dtype(dtype))
        return x_data, y_data

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = AbaloneDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return mean_remove_y(Ytr, Yts)

    def dset_name(self):
        return self._dset_name


class CaspDataset(BaseDataset):
    file_name = '/data/DATASETS/misc/casp.hdf5'
    _dset_name = 'casp'
    _default_train_frac = 0.7

    @staticmethod
    def read_data(dtype):
        with h5py.File(CaspDataset.file_name, 'r') as h5py_file:
            X = np.array(h5py_file['X'], dtype=as_np_dtype(dtype))
            Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype))
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = CaspDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return mean_remove_y(Ytr, Yts)

    def dset_name(self):
        return self._dset_name


class BlogFeedbackDataset(BaseDataset):
    file_name = '/data/DATASETS/misc/BlogFeedback.hdf5'
    _dset_name = 'blog-feedback'
    _num_train_samples = 52397

    @staticmethod
    def read_data(dtype):
        with h5py.File(BlogFeedbackDataset.file_name, 'r') as h5py_file:
            X_train = np.array(h5py_file['X_train'], dtype=as_np_dtype(dtype))
            Y_train = np.array(h5py_file['Y_train'], dtype=as_np_dtype(dtype))
            X_test = np.array(h5py_file['X_test'], dtype=as_np_dtype(dtype))
            Y_test = np.array(h5py_file['Y_test'], dtype=as_np_dtype(dtype))
        X = np.concatenate([X_train, X_test], axis=0)
        Y = np.concatenate([Y_train, Y_test], axis=0)
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            idx_tr = np.arange(BlogFeedbackDataset._num_train_samples)
            idx_ts = np.arange(BlogFeedbackDataset._num_train_samples, X.shape[0])
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mXtr = Xtr.mean(axis=0, keepdims=True, dtype=np.float64).astype(Xtr.dtype)
        sXtr = Xtr.std(axis=0, keepdims=True, dtype=np.float64, ddof=1).astype(Xtr.dtype)
        sXtr[sXtr == 0] = 1.0

        Xtr -= mXtr
        Xtr /= sXtr
        Xts -= mXtr
        Xts /= sXtr

        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Ytr.reshape(-1, 1), Yts.reshape(-1, 1), {}

    def dset_name(self):
        return self._dset_name


class CovTypeDataset(BaseDataset):
    file_name = '/data/DATASETS/misc/covtype_binary.hdf5'
    _dset_name = 'covtype'
    _default_train_frac = 0.7

    @staticmethod
    def read_data(dtype):
        with h5py.File(CovTypeDataset.file_name, 'r') as h5py_file:
            X = np.array(h5py_file['X'], dtype=as_np_dtype(dtype))
            Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype))
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = CovTypeDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        # Convert from 1, 2 to -1, +1
        Ytr = (Ytr - 1) * 2 - 1
        Yts = (Yts - 1) * 2 - 1
        return Ytr.reshape(-1, 1), Yts.reshape(-1, 1), {}

    def dset_name(self):
        return self._dset_name


class Ijcnn1Dataset(BaseDataset):
    file_name = '/data/DATASETS/misc/ijcnn1.hdf5'
    _dset_name = 'ijcnn1'
    _train_num = 49990

    @staticmethod
    def read_data(dtype):
        with h5py.File(Ijcnn1Dataset.file_name, "r") as f:
            Xtr = np.array(f["X_train"], dtype=as_np_dtype(dtype))
            Ytr = np.array(f["Y_train"], dtype=as_np_dtype(dtype))
            Xts = np.array(f["X_test"], dtype=as_np_dtype(dtype))
            Yts = np.array(f["Y_test"], dtype=as_np_dtype(dtype))
        return np.concatenate((Xtr, Xts), 0), np.concatenate((Ytr, Yts), 0)

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            idx_tr = np.arange(0, Ijcnn1Dataset._train_num)
            idx_ts = np.arange(Ijcnn1Dataset._train_num, X.shape[0])
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}  # Data already standardized

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Ytr.reshape(-1, 1), Yts.reshape(-1, 1), {}

    def dset_name(self):
        return self._dset_name


""" Public API """

__LOADERS = {
    Dataset.TIMIT: TimitDataset(),
    Dataset.HIGGS: HiggsDataset(),
    Dataset.MILLIONSONGS: MillionSongsDataset(),
    Dataset.TAXI: NycTaxiDataset(),
    Dataset.YELP: YelpDataset(),
    Dataset.FLIGHTS: FlightsDataset(),
    Dataset.SUSY: SusyDataset(),
    Dataset.MNIST: MnistDataset(),
    Dataset.FLIGHTS_CLS: FlightsClsDataset(),
    Dataset.SVHN: SVHNDataset(),
    Dataset.MNIST_SMALL: MnistSmallDataset(),
    Dataset.CIFAR10: CIFAR10Dataset(),
    Dataset.HOHIGGS: SmallHiggsDataset(),
    Dataset.ICTUS: IctusDataset(),
    Dataset.SYNTH01NOISE: SyntheticDataset(),
    Dataset.CHIET: ChietDataset(),
    Dataset.ENERGY: EnergyDataset(),
    Dataset.BOSTON: BostonDataset(),
    Dataset.PROTEIN: ProteinDataset(),
    Dataset.KIN40K: Kin40kDataset(),
    Dataset.CODRNA: CodRnaDataset(),
    Dataset.SVMGUIDE1: SvmGuide1Dataset(),
    Dataset.PHISHING: PhishingDataset(),
    Dataset.SPACEGA: SpaceGaDataset(),
    Dataset.CADATA: CadataDataset(),
    Dataset.MG: MgDataset(),
    Dataset.CPUSMALL: CpuSmallDataset(),
    Dataset.ABALONE: AbaloneDataset(),
    Dataset.CASP: CaspDataset(),
    Dataset.BLOGFEEDBACK: BlogFeedbackDataset(),
    Dataset.COVTYPE: CovTypeDataset(),
    Dataset.IJCNN1: Ijcnn1Dataset(),
}


def get_load_fn(dset: Dataset):
    try:
        return __LOADERS[dset].load_data
    except KeyError:
        raise KeyError(dset, f"No loader function found for dataset {dset}.")


def get_cv_fn(dset: Dataset):
    try:
        return __LOADERS[dset].load_data_cv
    except KeyError:
        raise KeyError(dset, f"No CV-loader function found for dataset {dset}.")
