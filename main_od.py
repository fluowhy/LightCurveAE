import argparse
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

import torch
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from models import GRUAE
from models import LSTMAE

from utils import load_json
from utils import load_yaml
from utils import WMSELoss
from utils import count_parameters
from utils import make_dir
from utils import seed_everything
from utils import plot_loss
from utils import od_metrics
from utils import save_yaml

from datasets import get_data_loaders


def learning_rate_update(lr, max_lr, err_1, err_2, beta_r, beta_e):
    if err_2 > 1.01 * err_1:
        new_lr = beta_r * lr
    elif (err_2 < err_1) and (lr < max_lr):
        new_lr = beta_e * lr
    else:
        new_lr = lr
    return new_lr


class Model(object):
    def __init__(self, args):
        self.args = args
        self.device = self.args["d"]
        if args["arch"] == "gru":
            self.model = GRUAE(args["nin"], args["nh"], args["nl"], args["nout"], args["nlayers"], args["do"])
        elif args["arch"] == "lstm":
            self.model = LSTMAE(args["nin"], args["nh"], args["nl"], args["nout"], args["nlayers"], args["do"])        
        self.model.to(device)
        print("model params {}".format(count_parameters(self.model)))
        log_path = "logs/autoencoder"
        if os.path.exists(log_path) and os.path.isdir(log_path):
            shutil.rmtree(log_path)
        # self.writer = SummaryWriter(log_path)
        self.wmse = WMSELoss(nc=2)
        self.best_loss = np.inf

    def train_model(self, data_loader, clip_value=1.):
        self.model.train()
        train_loss = 0
        for idx, batch in tqdm(enumerate(data_loader)):
            self.optimizer.zero_grad()
            x, y, seq_len = batch
            y, seq_len = y.to(self.device), seq_len.to(self.device)     
            x_pred, h = self.model(x, seq_len.long())
            loss = self.wmse(x, x_pred, seq_len).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= (idx + 1)
        return train_loss

    def eval_model(self, data_loader):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data_loader)):
                x, y, seq_len = batch
                y, seq_len = y.to(self.device), seq_len.to(self.device)
                x_pred, h = self.model(x, seq_len.long())
                loss = self.wmse(x, x_pred, seq_len).mean()
                eval_loss += loss.item()
        eval_loss /= (idx + 1)
        return eval_loss

    def fit(self, train_loader, val_loader):
        lr = self.args["lr"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args["wd"], amsgrad=True)
        loss = np.zeros((self.args["e"], 2))
        for epoch in range(self.args["e"]):
            template = "Epoch {} train loss {:.4f} val loss {:.4f}"
            train_loss = self.train_model(train_loader)
            val_loss = self.eval_model(val_loader)
            loss[epoch] = (train_loss, val_loss)
            # self.writer.add_scalars("recon", {"train": train_loss, "val": val_loss}, global_step=epoch)
            print(template.format(epoch, train_loss, val_loss))
            if val_loss < self.best_loss:
                self.best_model = self.model.state_dict()
                self.best_loss = val_loss
            lr = learning_rate_update(lr, self.args["max_lr"], loss[-1][0], loss[-2][0], self.args["beta_r"], self.args["beta_e"])
            print(lr)
            for g in self.optimizer.param_groups:
                g["lr"] = lr
        self.last_model = self.model.state_dict()
        return loss, self.best_model, self.last_model

    def evaluate(self, testset, valset, outlier_class):
        x_test = torch.nn.utils.rnn.pad_sequence(testset.x, padding_value=0., batch_first=True)
        seq_len = torch.tensor(testset.sl, dtype=torch.float, device=self.device)
        x_val = torch.nn.utils.rnn.pad_sequence(valset.x, padding_value=0., batch_first=True)
        seq_len_val = torch.tensor(valset.sl, dtype=torch.float, device=self.device)

        self.model.load_state_dict(self.best_model)
        self.model.eval()
        with torch.no_grad():
            x_pred, h = self.model(x_test, seq_len.long())
            x_pred_val, h_val = self.model(x_val, seq_len_val.long())
        recon_error = self.wmse(x_test, x_pred, seq_len)
        recon_error_val = self.wmse(x_val, x_pred_val, seq_len_val)
        x_pred = x_pred.cpu().numpy()
        h = h.cpu().numpy()
        self.scores = recon_error.cpu().numpy()
        self.scores_val = recon_error_val.cpu().numpy()
        
        y_test = np.array(testset.y)
        self.targets = np.ones(len(y_test))
        for oc in outlier_class:
            self.targets[y_test == oc] = 0
        # self.average_precision = (self.targets == 0).sum() / len(self.targets)
        self.aucpr, _, _, self.aucroc, _, _ = od_metrics(self.scores, self.targets, split=True, n_splits=100)
        return

    def plot_precision_recall(self, savename):
        plt.clf()
        plt.title("AUCPR: {:.4f}".format(self.aucpr))
        plt.plot(self.recall, self.precision, color="red")
        plt.axhline(self.average_precision, color="black", linestyle="--")
        plt.grid()
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(savename, dpi=200)
        return

    def plot_roc(self, savename):
        plt.clf()
        plt.title("AUCROC: {:.4f}".format(self.aucroc))
        plt.plot(self.fpr, self.tpr, color="red")
        aux = np.linspace(0, 1, 10)
        plt.plot(aux, aux, color="black", linestyle="--")
        plt.grid()
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(savename, dpi=200)
        return

    def plot_scores(self, savename, nbins=20):
        xmin = np.min(self.scores)  # , np.min(self.scores_val))
        xmax = np.max(self.scores)  # , np.max(self.scores_val))
        bins = np.linspace(xmin, xmax, nbins)
        plt.clf()
        plt.hist(self.scores_val, bins=bins, color="black", label="val", histtype="step")
        plt.hist(self.scores[self.targets == 1], bins=bins, color="navy", label="inlier", histtype="step")
        plt.hist(self.scores[self.targets == 0], bins=bins, color="red", label="outlier", histtype="step")
        plt.grid()
        plt.yscale("log")
        plt.legend()
        plt.xlabel("reconstruction error")
        plt.ylabel("counts")
        plt.tight_layout()
        plt.savefig(savename, dpi=200)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autoencoder")
    parser.add_argument('--e', type=int, default=2, help="epochs (default 2)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="linear",
        choices=["asas", "linear", "macho", "asas_sn", "toy", "ztf_transient", "ztf_stochastic", "ztf_periodic"],
        help="dataset name (default linear)"
    )
    parser.add_argument('--config', type=int, default=0, help="config number (default 0)")
    parser.add_argument('--device', type=str, default="cpu", help="device (default cpu)")
    parser.add_argument('--oc', type=int, default=0, help="outlier class (default 0)")
    args = parser.parse_args()
    print(args)

    config = load_yaml("config/config_0.yaml")
    fold = config["fold"]
    bs = config["bs"]
    device = args.device
    config["d"] = device
    arch = config["arch"]
    config["e"] = args.e

    seed_everything()

    make_dir("figures")
    make_dir("files")
    make_dir("models")

    make_dir("figures/od")
    make_dir("files/od")
    make_dir("models/od")

    make_dir("figures/od/{}".format(args.dataset))
    make_dir("files/od/{}".format(args.dataset))
    make_dir("models/od/{}".format(args.dataset))

    make_dir("figures/od/{}/config_{}".format(args.dataset, args.config))
    make_dir("files/od/{}/config_{}".format(args.dataset, args.config))
    make_dir("models/od/{}/config_{}".format(args.dataset, args.config))

    models_dir = f"models/od/{args.dataset}/config_{args.config}/oc_{args.oc}"
    figures_dir = f"figures/od/{args.dataset}/config_{args.config}/oc_{args.oc}"
    files_dir = f"files/od/{args.dataset}/config_{args.config}/oc_{args.oc}"

    make_dir(figures_dir)
    make_dir(files_dir)
    make_dir(models_dir)

    # dataset = LightCurveDataset(args.dataset, fold=True, bs=bs, device=device, eval=True)
    if args.dataset == "toy":
        dataset = ToyDataset(args, val_size=0.1, sl=64)
        outlier_class = [3, 4]
    elif args.dataset == "asas_sn":
        trainloader, valloader, testloader = get_data_loaders(args.dataset, config["bs"], device)
        outlier_class = [8]
    elif "ztf" in args.dataset:
        trainloader, valloader, testloader = get_data_loaders(args.dataset, config["bs"], device)
        outlier_labels = load_json("../datasets/ztf/cl/transient/lab2out.json")
        outlier_labels = [key for key in outlier_labels if outlier_labels[key] == "outlier"]
        lab2idx = load_json("../datasets/ztf/cl/transient/lab2idx.json")
        outlier_class = [int(lab2idx[lab]) for lab in outlier_labels]
    elif args.dataset == "asas":
        outlier_class = [args.oc]
        trainloader, valloader, testloader = get_data_loaders(args.dataset, config["bs"], device, args.oc)
    elif args.dataset == "linear":
        outlier_class = [args.oc]
        trainloader, valloader, testloader = get_data_loaders(args.dataset, config["bs"], device, args.oc)

    config["nin"] = trainloader.dataset.x[0].shape[1]

    print(config)
    autoencoder = Model(config)
    loss, best_model, last_model = autoencoder.fit(trainloader, valloader)
    torch.save(best_model, f"{models_dir}/best.pth")
    torch.save(last_model, f"{models_dir}last.pth")
    autoencoder.evaluate(testloader.dataset, valloader.dataset, outlier_class)
    plot_loss(loss, f"{figures_dir}/loss.png")
    # autoencoder.plot_precision_recall("figures/od/{}/config_{}/precision_recall.png".format(args.dataset, args.config))
    # autoencoder.plot_roc("figures/od/{}/config_{}/roc.png".format(args.dataset, args.config))
    autoencoder.plot_scores(f"{figures_dir}/score.png", nbins=50)
    data = dict(
        aucpr=float(np.mean(autoencoder.aucpr)),
        aucpr_std=float(np.std(autoencoder.aucpr)),
        aucroc=float(np.mean(autoencoder.aucroc)),
        aucroc_std=float(np.std(autoencoder.aucroc)),
    )
    save_yaml(data, f"{files_dir}/metrics.yaml")
