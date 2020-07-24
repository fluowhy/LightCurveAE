import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
import shutil
import json
import matplotlib.pyplot as plt
import pdb


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


class WMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-10):
        super(WMSELoss, self).__init__()
        self.eps = eps

    def forward(self, x, x_pred, seq_len):
        # x_pred -> (mu, logvar)
        mask = (x[:, :, 2] != 0) * 1.
        wmse = (((x_pred - x[:, :, 1]) / (x[:, :, 2] + self.eps)).pow(2) * mask).sum(dim=- 1) / seq_len
        return wmse


class MyDataset(Dataset):
    def __init__(self, x, z, device="cpu"):
        self.n, _, _ = x.shape  # rnn
        self.x = torch.tensor(x, dtype=torch.float, device=device)
        self.z = torch.tensor(z, dtype=torch.float, device=device)

    def __getitem__(self, index):
        return self.x[index], self.z[index]

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
    