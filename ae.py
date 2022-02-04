import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from models import GRUAE
from models import LSTMAE

from utils import count_parameters
from utils import WMSELoss
from utils import od_metrics


class Model(object):
    def __init__(self, args):
        self.args = args
        self.device = self.args["d"]
        if args["arch"] == "gru":
            self.model = GRUAE(args["nin"], args["nh"], args["nl"], args["nout"], args["nlayers"], args["do"])
        elif args["arch"] == "lstm":
            self.model = LSTMAE(args["nin"], args["nh"], args["nl"], args["nout"], args["nlayers"], args["do"])        
        self.model.to(self.device)
        print("model params {}".format(count_parameters(self.model)))
        # log_path = "logs/autoencoder"
        # if os.path.exists(log_path) and os.path.isdir(log_path):
            # shutil.rmtree(log_path)
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
            # lr = learning_rate_update(lr, self.args["max_lr"], loss[-1][0], loss[-2][0], self.args["beta_r"], self.args["beta_e"])
            # print(lr)
            # for g in self.optimizer.param_groups:
                # g["lr"] = lr
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
        return self.scores

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