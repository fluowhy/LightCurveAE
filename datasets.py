import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import platform

from utils import get_ztf_data
from utils import get_asas_data
from utils import get_linear_data
from utils import make_dir
from utils import save_json
from utils import MyDataset
from utils import get_asas_sn_data


def pad_sequence_with_lengths(data):
    x = torch.nn.utils.rnn.pad_sequence([d[0] for d in data], padding_value=0., batch_first=True)
    y = torch.tensor([d[1] for d in data], dtype=torch.long)
    seq_len = torch.tensor([d[2] for d in data], dtype=torch.long)
    return x, y, seq_len


def apply_normalization(x):
    normalize_time(x, offset=True)
    normalize(x)
    return


def normalize_time(x, offset=False):
    if offset:
        for xi in x:
            xi[:, 0] = np.log10(xi[:, 0] - xi[:, 0].min() + 1e-10)
            xi[0, 0] = 0
    else:
        for xi in x:
            xi[:, 0] = np.log10(xi[:, 0] + 1e-10)
    return


def normalize(x):
    for xi in x:
        mean = xi[:, 1].mean()
        std = xi[:, 1].std()
        xi[:, 1] = (xi[:, 1] - mean) / std
        xi[:, 2] = xi[:, 2] / std
    return


def read_and_normalize_data(data_path):
    x = np.load(data_path, allow_pickle=True)
    x_train = x["x_train"]
    x_val = x["x_val"]
    y_train = x["y_train"]
    y_val = x["y_val"]
    x_test = x["x_test"]
    y_test = x["y_test"]
    apply_normalization(x_train)
    apply_normalization(x_val)
    apply_normalization(x_test)
    return x_train, x_val, x_test, y_train, y_val, y_test


def load_data(device="cpu"):
    if platform.system() == "Windows":
        data_path = "C:/Users/mauricio/Documents/datasets/ztf/train_data_band_1.npz"
    elif platform.system() == "Linux":
        data_path = "/home/mromero/datasets/ztf/train_data_band_1.npz"
    x_train, x_val, x_test, y_train, y_val, y_test = read_and_normalize_data(data_path)
    trainset = MyDataset(x_train, y_train, device)
    valset = MyDataset(x_val, y_val, device)
    testset = MyDataset(x_test, y_test, device)        
    return trainset, valset, testset


def get_data_loaders(dataset, batch_size, device, oc=None):
    if dataset == "asas_sn":
        x_train, x_val, x_test, y_train, y_val, y_test = get_asas_sn_data()
    elif "ztf" in dataset:
        x_train, x_val, x_test, y_train, y_val, y_test = get_ztf_data(dataset)
    elif dataset == "asas":
        x_train, x_val, x_test, y_train, y_val, y_test = get_asas_data(oc)
    elif dataset == "linear":
        x_train, x_val, x_test, y_train, y_val, y_test = get_linear_data(oc)
    trainset, valset, testset = get_datasets(x_train, x_val, x_test, y_train, y_val, y_test, device)
    trainloader = DataLoader(trainset, batch_size=int(batch_size), shuffle=True, collate_fn=pad_sequence_with_lengths)
    valloader = DataLoader(valset, batch_size=int(batch_size), shuffle=True, collate_fn=pad_sequence_with_lengths)
    testloader = DataLoader(testset, batch_size=int(batch_size), shuffle=False, collate_fn=pad_sequence_with_lengths)
    return trainloader, valloader, testloader


def load_asas_sn_data(device="cpu"):
    x_train, x_val, x_test, y_train, y_val, y_test = read_and_normalize_asas_sn_data()
    trainset = MyDataset(x_train, y_train, device)
    valset = MyDataset(x_val, y_val, device)
    testset = MyDataset(x_test, y_test, device)
    return trainset, valset, testset


def get_datasets(x_train, x_val, x_test, y_train, y_val, y_test, device):
    return MyDataset(x_train, y_train, device), MyDataset(x_val, y_val, device), MyDataset(x_test, y_test, device)


def read_and_normalize_asas_sn_data():
    asas_sn_pipeline = ASASSNPipeline()
    x = np.load("../datasets/asas_sn/train_data.npz", allow_pickle=True)
    x_train = x["x_train"]
    x_val = x["x_val"]
    x_test = x["x_test"]
    y_train = x["y_train"]
    y_val = x["y_val"]    
    y_test = x["y_test"]
    x_train = asas_sn_pipeline(x_train)
    x_val = asas_sn_pipeline(x_val)
    x_test = asas_sn_pipeline(x_test)
    import pdb
    pdb.set_trace()
    return x_train, x_val, x_test, y_train, y_val, y_test


class ASASSNPipeline():
    def __init__(self, eps=1e-10):
        self.eps = eps
        pass

    def normalize_lc(self, x):
        m, s = x[:, 1].mean(), x[:, 1].std()
        x[:, 1] = (x[:, 1] - m) / (s + self.eps)
        x[:, 2] = x[:, 2] / (s + self.eps)
        return
    
    def t2dt(self, x):
        dt = x[1:, 0] - x[:-1, 0]
        x[1:, 0] = dt
        x[0, 0] = 0.
        return
    
    def remove_dt_0(self, x):
        dt = x[1:, 0] - x[:-1, 0]
        mask = dt != 0
        mask = np.insert(mask, 0, True)
        return x[mask]

    def __call__(self, x):
        x = [self.remove_dt_0(xi) for xi in x]
        for xi in tqdm(x): self.normalize_lc(xi)
        for xi in tqdm(x): self.t2dt(xi)
        return x





def phase_fold(x, p, seq_len):
    x_copy = x.copy()
    t = x[:seq_len, 0]
    mag = x[:seq_len, 1]
    err = x[:seq_len, 2]    
    t = t - t[0]
    t = (t % p) / p
    index = np.argsort(t)
    t = t[index]
    mag = mag[index]
    err = err[index]
    x_copy[:seq_len, 0] = t
    x_copy[:seq_len, 1] = mag
    x_copy[:seq_len, 2] = err
    return x_copy


class LightCurveDataset(object):
    def __init__(self, name, fold, bs, device="cpu", val_size=0.2, eps=1e-10, eval=True):
        self.name = name
        self.fold = fold
        self.eps = eps
        self.val_size = val_size
        self.bs = bs
        self.device = device

        self.data_path = "processed_data/{}".format(name)
        make_dir("processed_data")
        make_dir(self.data_path)
        
        if not eval:
            self.load_data()
            self.fold_light_curves()
            self.normalize()
            self.compute_dt()
            self.time_series_features()
            self.train_val_split()
            self.save_processed_data()
        else:
            self.load_processed_data()
            self.compute_seq_len()

        self.define_datasets()

    def load_data(self):
        self.x = np.load("../datasets/{}/light_curves.npy".format(self.name))
        self.metadata = pd.read_csv("../datasets/{}/metadata.csv".format(self.name))
        return

    def fold_light_curves(self):
        self.x_folded = self.x.copy()
        for i in tqdm(range(len(self.x))):
            dfi = self.metadata.loc[i]
            self.x_folded[i] = phase_fold(self.x[i], dfi["p"], int(dfi["seq_len"]))
        return

    def normalize(self):
        mask = self.x[:, :, 2] != 0
        means = (self.metadata["mean"].values)[:, np.newaxis]
        stds = (self.metadata["std"].values)[:, np.newaxis]
        self.x[:, :, 1] = (self.x[:, :, 1] - means) / (stds + self.eps) * mask
        self.x[:, :, 2] = self.x[:, :, 2] / (stds + self.eps) * mask

        self.x_folded[:, :, 1] = (self.x_folded[:, :, 1] - means) / (stds + self.eps) * mask
        self.x_folded[:, :, 2] = self.x_folded[:, :, 2] / (stds + self.eps) * mask
        return

    def compute_dt(self):
        mask = self.x[:, :, 2] != 0
        self.x[:, 1:, 0] = self.x[:, 1:, 0] - self.x[:, :-1, 0]
        self.x[:, 0, 0] = 0
        self.x[:, :, 0] *= mask

        self.x_folded[:, 1:, 0] = self.x_folded[:, 1:, 0] - self.x_folded[:, :-1, 0]
        self.x_folded[:, 0, 0] = 0
        self.x_folded[:, :, 0] *= mask
        return

    def time_series_features(self):
        self.m = self.metadata["mean"].values
        self.s = self.metadata["std"].values
        self.p = self.metadata["p"].values
        self.p = np.log10(self.p)
        self.seq_len = self.metadata["seq_len"].values
        return

    def train_val_split(self):
        index = np.arange(len(self.x))
        lab2idx = {lab: i for i, lab in enumerate(self.metadata["label"].unique())}
        save_json(lab2idx, "{}/lab2idx.json".format(self.data_path))
        self.y  = [lab2idx[lab] for lab in self.metadata["label"].values]
        self.y = np.array(self.y)      
        train_idx, val_idx = train_test_split(index, test_size=self.val_size, stratify=self.y, shuffle=True)

        self.x_train = self.x[train_idx]
        self.x_train_folded = self.x_folded[train_idx]
        self.y_train = self.y[train_idx]
        self.m_train = self.m[train_idx]
        self.s_train = self.s[train_idx]
        self.p_train = self.p[train_idx]
        self.seq_len_train = self.seq_len[train_idx]

        self.x_val = self.x[val_idx]
        self.x_val_folded = self.x_folded[val_idx]
        self.y_val = self.y[val_idx]
        self.m_val = self.m[val_idx]
        self.s_val = self.s[val_idx]
        self.p_val = self.p[val_idx]
        self.seq_len_val = self.seq_len[val_idx]
        return

    def save_processed_data(self):
        np.save("{}/x_train.npy".format(self.data_path), self.x_train)
        np.save("{}/x_train_folded.npy".format(self.data_path), self.x_train_folded)
        np.save("{}/y_train.npy".format(self.data_path), self.y_train)
        np.save("{}/m_train.npy".format(self.data_path), self.m_train)
        np.save("{}/s_train.npy".format(self.data_path), self.s_train)
        np.save("{}/p_train.npy".format(self.data_path), self.p_train)
        
        np.save("{}/x_val.npy".format(self.data_path), self.x_val)
        np.save("{}/x_val_folded.npy".format(self.data_path), self.x_val_folded)
        np.save("{}/y_val.npy".format(self.data_path), self.y_val)
        np.save("{}/m_val.npy".format(self.data_path), self.m_val)
        np.save("{}/s_val.npy".format(self.data_path), self.s_val)
        np.save("{}/p_val.npy".format(self.data_path), self.p_val)
        return

    def load_processed_data(self):
        if self.fold:
            self.x_train = np.load("{}/x_train_folded.npy".format(self.data_path))
            self.p_train = np.load("{}/p_train.npy".format(self.data_path))
            self.x_val = np.load("{}/x_val_folded.npy".format(self.data_path))
            self.p_val = np.load("{}/p_val.npy".format(self.data_path))
        else:
            self.x_train = np.load("{}/x_train.npy".format(self.data_path))
            self.x_val = np.load("{}/x_val.npy".format(self.data_path))

        self.y_train = np.load("{}/y_train.npy".format(self.data_path))
        self.m_train = np.load("{}/m_train.npy".format(self.data_path))
        self.s_train = np.load("{}/s_train.npy".format(self.data_path))
        
        self.y_val = np.load("{}/y_val.npy".format(self.data_path))
        self.m_val = np.load("{}/m_val.npy".format(self.data_path))
        self.s_val = np.load("{}/s_val.npy".format(self.data_path))
        return

    def define_datasets(self):
        self.train_dataset = MyDataset(self.x_train, self.seq_len_train, device=self.device)
        self.val_dataset = MyDataset(self.x_val, self.seq_len_val, device=self.device)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.bs, shuffle=True)
        return

    def compute_seq_len(self):
        self.seq_len_train = (self.x_train[:, :, 2] != 0).sum(axis=1)
        self.seq_len_val = (self.x_val[:, :, 2] != 0).sum(axis=1)
        return


class ASASSNDataset(object):
    def __init__(self, fold, bs, device="cpu", val_size=0.2, eps=1e-10, eval=True):
        self.name = "asas_sn"
        self.fold = fold
        self.eps = eps
        self.val_size = val_size
        self.bs = bs
        self.device = device

        self.data_path = "processed_data/{}".format(self.name)
        make_dir("processed_data")
        make_dir(self.data_path)
        
        if not eval:
            self.load_data()
            self.x_train, self.x_train_folded, self.m_train, self.s_train = self.normalize(self.x_train, self.x_train_folded)
            self.x_test, self.x_test_folded, self.m_test, self.s_test = self.normalize(self.x_test, self.x_test_folded)
            self.x_val, self.x_val_folded, self.m_val, self.s_val = self.normalize(self.x_val, self.x_val_folded)
            self.x_train, self.x_train_folded = self.compute_dt(self.x_train, self.x_train_folded)
            self.x_test, self.x_test_folded = self.compute_dt(self.x_test, self.x_test_folded)
            self.x_val, self.x_val_folded = self.compute_dt(self.x_val, self.x_val_folded)
            self.seq_len_train, self.p_train = self.time_series_features(self.x_train, self.p_train)
            self.seq_len_test, self.p_test = self.time_series_features(self.x_test, self.p_test)
            self.seq_len_val, self.p_val = self.time_series_features(self.x_val, self.p_val)
            self.save_processed_data()
        else:
            self.load_processed_data()
            self.compute_seq_len()

        self.define_datasets()

    def load_data(self):
        self.x_train = np.load("../datasets/asas_sn/x_train.npy")
        self.x_train_folded = np.load("../datasets/asas_sn/pf_train.npy")
        self.y_train = np.load("../datasets/asas_sn/y_train.npy")
        self.p_train = np.load("../datasets/asas_sn/p_train.npy")

        self.x_test = np.load("../datasets/asas_sn/x_test.npy")
        self.x_test_folded = np.load("../datasets/asas_sn/pf_test.npy")
        self.y_test = np.load("../datasets/asas_sn/y_test.npy")
        self.p_test = np.load("../datasets/asas_sn/p_test.npy")

        self.x_val = np.load("../datasets/asas_sn/x_val.npy")
        self.x_val_folded = np.load("../datasets/asas_sn/pf_val.npy")
        self.y_val = np.load("../datasets/asas_sn/y_val.npy")
        self.p_val = np.load("../datasets/asas_sn/p_val.npy")
        return

    def normalize(self, x, x_folded):
        n = len(x)
        mask = x[:, :, 2] != 0
        seq_len = mask.sum(1)
        m = np.zeros(n)
        s = np.zeros(n)
        for i in range(n):
            m[i] = x[i, :seq_len[i], 1].mean()
            s[i] = x[i, :seq_len[i], 1].std()
        x[:, :, 1] = (x[:, :, 1] - m[:, np.newaxis]) / (s[:, np.newaxis] + self.eps) * mask
        x[:, :, 2] = x[:, :, 2] / (s[:, np.newaxis] + self.eps) * mask
        x_folded[:, :, 1] = (x_folded[:, :, 1] - m[:, np.newaxis]) / (s[:, np.newaxis] + self.eps) * mask
        x_folded[:, :, 2] = x_folded[:, :, 2] / (s[:, np.newaxis] + self.eps) * mask
        return x, x_folded, m, s

    def compute_dt(self, x, x_folded):
        mask = x[:, :, 2] != 0
        x[:, 1:, 0] = x[:, 1:, 0] - x[:, :-1, 0]
        x[:, 0, 0] = 0
        x[:, :, 0] *= mask

        x_folded[:, 1:, 0] = x_folded[:, 1:, 0] - x_folded[:, :-1, 0]
        x_folded[:, 0, 0] = 0
        x_folded[:, :, 0] *= mask
        return x, x_folded

    def time_series_features(self, x, p):
        seq_len = (x[:, :, 2] != 0).sum(1)
        return seq_len, np.log10(p)

    def save_processed_data(self):
        np.save("{}/x_train.npy".format(self.data_path), self.x_train)
        np.save("{}/x_train_folded.npy".format(self.data_path), self.x_train_folded)
        np.save("{}/y_train.npy".format(self.data_path), self.y_train)
        np.save("{}/m_train.npy".format(self.data_path), self.m_train)
        np.save("{}/s_train.npy".format(self.data_path), self.s_train)
        np.save("{}/p_train.npy".format(self.data_path), self.p_train)
        
        np.save("{}/x_val.npy".format(self.data_path), self.x_val)
        np.save("{}/x_val_folded.npy".format(self.data_path), self.x_val_folded)
        np.save("{}/y_val.npy".format(self.data_path), self.y_val)
        np.save("{}/m_val.npy".format(self.data_path), self.m_val)
        np.save("{}/s_val.npy".format(self.data_path), self.s_val)
        np.save("{}/p_val.npy".format(self.data_path), self.p_val)

        np.save("{}/x_test.npy".format(self.data_path), self.x_test)
        np.save("{}/x_test_folded.npy".format(self.data_path), self.x_test_folded)
        np.save("{}/y_test.npy".format(self.data_path), self.y_test)
        np.save("{}/m_test.npy".format(self.data_path), self.m_test)
        np.save("{}/s_test.npy".format(self.data_path), self.s_test)
        np.save("{}/p_test.npy".format(self.data_path), self.p_test)
        return

    def load_processed_data(self):
        if self.fold:
            self.x_train = np.load("{}/x_train_folded.npy".format(self.data_path))
            self.p_train = np.load("{}/p_train.npy".format(self.data_path))
            self.x_val = np.load("{}/x_val_folded.npy".format(self.data_path))
            self.p_val = np.load("{}/p_val.npy".format(self.data_path))
            self.x_test = np.load("{}/x_test_folded.npy".format(self.data_path))
            self.p_test = np.load("{}/p_test.npy".format(self.data_path))
        else:
            self.x_train = np.load("{}/x_train.npy".format(self.data_path))
            self.x_val = np.load("{}/x_val.npy".format(self.data_path))
            self.x_test = np.load("{}/x_test.npy".format(self.data_path))

        self.y_train = np.load("{}/y_train.npy".format(self.data_path))
        self.m_train = np.load("{}/m_train.npy".format(self.data_path))
        self.s_train = np.load("{}/s_train.npy".format(self.data_path))
        
        self.y_val = np.load("{}/y_val.npy".format(self.data_path))
        self.m_val = np.load("{}/m_val.npy".format(self.data_path))
        self.s_val = np.load("{}/s_val.npy".format(self.data_path))

        self.y_test = np.load("{}/y_test.npy".format(self.data_path))
        self.m_test = np.load("{}/m_test.npy".format(self.data_path))
        self.s_test = np.load("{}/s_test.npy".format(self.data_path))
        return

    def define_datasets(self):
        # balancing
        labs, counts = np.unique(self.y_train, return_counts=True)
        mask = labs != -99
        weights = 1 / counts[mask]
        weights /= 2 * weights.sum()
        weights = np.insert(weights, 0, 0.5)

        sample_weight = np.zeros(len(self.y_train))
        for i, lab in enumerate(labs):
            mask = self.y_train == lab
            sample_weight[mask] = weights[i]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weight, len(sample_weight))

        self.train_dataset = MyAugDataset(self.x_train, self.y_train, self.m_train, self.s_train, self.seq_len_train, device=self.device)
        self.val_dataset = MyAugDataset(self.x_val, self.y_val, self.m_val, self.s_val, self.seq_len_val, device=self.device)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.bs, sampler=sampler)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.bs, shuffle=True)
        return

    def compute_seq_len(self):
        self.seq_len_train = (self.x_train[:, :, 2] != 0).sum(axis=1)
        self.seq_len_val = (self.x_val[:, :, 2] != 0).sum(axis=1)
        self.seq_len_test = (self.x_test[:, :, 2] != 0).sum(axis=1)
        return



