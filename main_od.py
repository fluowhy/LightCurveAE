import argparse
import torch
import numpy as np

from utils import load_json
from utils import load_yaml
from utils import make_dir
from utils import seed_everything
from utils import plot_loss
from utils import save_yaml

from ae import Model

from datasets import get_data_loaders


def learning_rate_update(lr, max_lr, err_1, err_2, beta_r, beta_e):
    if err_2 > 1.01 * err_1:
        new_lr = beta_r * lr
    elif (err_2 < err_1) and (lr < max_lr):
        new_lr = beta_e * lr
    else:
        new_lr = lr
    return new_lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="autoencoder")
    parser.add_argument('--e', type=int, default=2, help="epochs (default 2)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="asas",
        choices=["asas", "linear", "macho", "asas_sn", "toy", "ztf_transient", "ztf_stochastic", "ztf_periodic"],
        help="dataset name (default asas)"
    )
    parser.add_argument('--config', type=int, default=0, help="config number (default 0)")
    parser.add_argument('--device', type=str, default="cpu", help="device (default cpu)")
    parser.add_argument('--oc', type=int, default=0, help="outlier class (default 0)")
    args = parser.parse_args()
    print(args)

    config = load_yaml("config/config_0.yaml")
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
    scores = autoencoder.evaluate(testloader.dataset, valloader.dataset, outlier_class)
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
    np.savez_compressed(f"{files_dir}/scores.npz", scores=scores)
