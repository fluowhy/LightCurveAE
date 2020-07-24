import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import *
from utils import *
from datasets import LightCurveDataset


class Model(object):
    def __init__(self, args):
        self.args = args
        if args.arch == "gru":
            self.model = GRUAE(args.nin, args.nh, args.nl, args.nout, args.nlayers, args.do)
        elif args.arch == "lstm":
            self.model = LSTMAE(args.nin, args.nh, args.nl, args.nout, args.nlayers, args.do)        
        self.model.to(args.d)
        print("model params {}".format(count_parameters(self.model)))
        log_path = "logs/autoencoder"
        if os.path.exists(log_path) and os.path.isdir(log_path):
            shutil.rmtree(log_path)
        self.writer = SummaryWriter(log_path)
        self.wmse = WMSELoss()
        self.best_loss = np.inf

    def train_model(self, data_loader, clip_value=1.):
        self.model.train()
        train_loss = 0
        for idx, batch in tqdm(enumerate(data_loader)):
            self.optimizer.zero_grad()
            x, seq_len = batch
            # x = x.to(self.args.d)
            # seq_len = seq_len.to(self.args.d)            
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
                x, seq_len = batch
                # x = x.to(self.args.d)
                # seq_len = seq_len.to(self.args.d)
                x_pred, h = self.model(x, seq_len.long())
                loss = self.wmse(x, x_pred, seq_len).mean()
                eval_loss += loss.item()
        eval_loss /= (idx + 1)
        return eval_loss

    def fit(self, train_loader, val_loader, args):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
        loss = np.zeros((args.e, 2))
        for epoch in range(args.e):
            template = "Epoch {} train loss {:.4f} val loss {:.4f}"
            train_loss = self.train_model(train_loader)
            val_loss = self.eval_model(val_loader)
            loss[epoch] = (train_loss, val_loss)
            self.writer.add_scalars("recon", {"train": train_loss, "val": val_loss}, global_step=epoch)
            print(template.format(epoch, train_loss, val_loss))
            if val_loss < self.best_loss:
                self.best_model = self.model.state_dict()
                self.best_loss = val_loss
        torch.save(self.best_model, "models/{}.pth".format(self.args.arch))
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autoencoder")
    parser.add_argument('--bs', type=int, default=128, help="batch size (default 128)")
    parser.add_argument('--e', type=int, default=2, help="epochs (default 2)")
    parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default 2e-4)")
    parser.add_argument('--nout', type=int, default=1, help="output size (default 1)")
    parser.add_argument('--nh', type=int, default=2, help="hidden size (default 2)")
    parser.add_argument('--nl', type=int, default=2, help="hidden size (default 2)")
    parser.add_argument('--nlayers', type=int, default=1, help="number of hidden layers (default 1)")
    parser.add_argument("--do", type=float, default=0., help="dropout value (default 0)")
    parser.add_argument("--wd", type=float, default=0., help="L2 reg value (default 0)")
    parser.add_argument("--arch", type=str, default="gru", choices=["gru", "lstm"], help="rnn architecture (default gru)")
    args = parser.parse_args()
    print(args)

    seed_everything()

    dataset = LightCurveDataset("linear", fold=True, bs=args.bs, device=args.d, eval=True)  
    args.nin = dataset.x_train.shape[2]

    autoencoder = Model(args)
    loss = autoencoder.fit(dataset.train_dataloader, dataset.val_dataloader, args)
    plot_loss(loss, "figures/{}_loss.png".format(args.arch))
