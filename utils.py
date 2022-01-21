import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pandas as pd


def normalize_mag(x):
    for xi in x:
        m, s = xi[:, 1].mean(), xi[:, 1].std()
        xi[:, 1] = (xi[:, 1] - m) / s
        xi[:, 2] = xi[:, 2] / s
    return


def remove_time_offset(x):
    for xi in x:
        xi[:, 0] = xi[:, 0] -  xi[:, 0].min()
    return


def get_ztf_data(dataset_name):
    family = dataset_name.split('_')[1]
    data = np.load("../datasets/ztf/cl/{}/light_curves.npz".format(family), allow_pickle=True)["light_curves"]
    df = pd.read_csv("../datasets/ztf/cl/{}/metadata_pro1.csv".format(family))
    remove_time_offset(data)
    normalize_mag(data)
    mask = df["split2"] == "train"
    x_train = [dat for dat, ma in zip(data, mask) if ma]
    y_train = df["label"][mask].values
    mask = df["split2"] == "val"
    x_val = [dat for dat, ma in zip(data, mask) if ma]
    y_val = df["label"][mask].values
    mask = df["split2"] == "test"
    x_test = [dat for dat, ma in zip(data, mask) if ma]
    y_test = df["label"][mask].values
    return x_train, x_val, x_test, y_train, y_val, y_test


def od_metrics(scores, y, split=False, n_splits=None):
    aucpr = list()
    aucroc = list()
    scores_in = scores[y == 1]
    scores_out = scores[y == 0]
    if split:
        _, counts = np.unique(y, return_counts=True)
        if counts[0] < counts[1]:
            new_y = np.concatenate((np.ones(counts[0]), np.zeros(counts[0])))            
            for i in range(n_splits):
                new_scores = np.random.choice(scores_in, counts[0], replace=False)
                cat_scores = np.concatenate((new_scores, scores_out))
                precision, recall, _ = precision_recall_curve(new_y, cat_scores, pos_label=0)
                fpr, tpr, _ = roc_curve(new_y, cat_scores, pos_label=0)
                aucpr.append(auc(recall, precision))
                aucroc.append(auc(fpr, tpr))
        else:
            new_y = np.concatenate((np.ones(counts[1]), np.zeros(counts[1])))
            for i in range(n_splits):
                new_scores = np.random.choice(scores_out, counts[1], replace=False)
                cat_scores = np.concatenate((scores_in, new_scores))
                precision, recall, _ = precision_recall_curve(new_y, cat_scores, pos_label=0)
                fpr, tpr, _ = roc_curve(new_y, cat_scores, pos_label=0)
                aucpr.append(auc(recall, precision))
                aucroc.append(auc(fpr, tpr))
        return aucpr, None, None, aucroc, None, None
    else:
        precision, recall, _ = precision_recall_curve(y, scores, pos_label=0)
        fpr, tpr, _ = roc_curve(y, scores, pos_label=0)
        aucpr = auc(recall, precision)
        aucroc = auc(fpr, tpr)
        return aucpr, precision, recall, aucroc, fpr, tpr


def save_yaml(data, savename):
    with open(savename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    return


def load_yaml(filename):
    with open(filename, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


def plot_confusion_matrix(cm, labels, title, savename, normalize=False, dpi=200):
    nc = len(labels)
    plt.clf()
    fig, ax = plt.subplots()
    if normalize:
        cm = cm / cm.sum(1)
        vmax = 1
        my_format = "{:.2f}"
    else:
        vmax = cm.max()
        my_format = "{:.0f}"
    thr = vmax * 0.5
    im = ax.imshow(cm, cmap="YlGn", aspect="equal", vmin=0, vmax=vmax)
    for i in range(nc):
        for j in range(nc):
            if cm[i, j] > thr:
                color = "white"
            else:
                color = "black"
            text = ax.text(j, i, my_format.format(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
    plt.title(title)
    plt.xticks(np.arange(nc), labels, rotation=45)
    plt.yticks(np.arange(nc), labels)
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(savename, dpi=dpi)
    return


def plot_loss(loss, savename, dpi=200):
    epochs = np.arange(len(loss))
    plt.clf()
    plt.plot(epochs, loss[:, 0], color="navy", label="train")
    plt.plot(epochs, loss[:, 1], color="red", label="val")
    plt.xlabel("epochs")
    plt.ylabel("reconstruction error")
    plt.grid()
    plt.savefig(savename, dpi=dpi)
    return


def save_json(data, savename):
    with open(savename, "w") as fp:
        json.dump(data, fp)


def load_json(filename):
    with open(filename, "r") as fp:
        data = json.load(fp)
    return data


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return


class MyAugDataset(Dataset):
    def __init__(self, x, y, m, s, z, device="cpu"):
        self.n, _, _ = x.shape
        self.x = torch.tensor(x, dtype=torch.float, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)
        self.m = torch.tensor(m, dtype=torch.float, device=device)
        self.s = torch.tensor(s, dtype=torch.float, device=device)
        self.z = torch.tensor(z, dtype=torch.float, device=device)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.m[index], self.s[index], self.z[index]

    def __len__(self):
        return self.n


class WMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-10, nc=3):
        super(WMSELoss, self).__init__()
        self.eps = eps
        self.loss_fun = self.wmse_3c if nc == 3 else self.wmse_2c

    def wmse_3c(self, x, x_pred, seq_len):
        mask = (x[:, :, 2] != 0) * 1.
        wmse = (((x_pred - x[:, :, 1]) / (x[:, :, 2] + self.eps)).pow(2) * mask).sum(dim=- 1) / seq_len
        return wmse

    def wmse_2c(self, x, x_pred, seq_len):
        mask = (x[:, :, 0] != 0) * 1.
        wmse = ((x_pred - x[:, :, 1]).pow(2) * mask).sum(dim=- 1) / seq_len
        return wmse

    def forward(self, x, x_pred, seq_len):
        wmse = self.loss_fun(x, x_pred, seq_len)
        return wmse


class MyDataset(Dataset):
    def __init__(self, x, y, device="cpu"):
        self.n = len(x)
        self.sl = [len(xi) for xi in x]
        self.x = [torch.tensor(xi, dtype=torch.float, device=device) for xi in x]
        self.y = y
        self.device = device

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.sl[index]

    def __len__(self):
        return self.n


def seed_everything(seed=1234):
    """
    Author: Benjamin Minixhofer
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return


def count_parameters(model):
    # TODO: add docstring
    """
    Parameters
    ----------
    model
    Returns
    -------
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    