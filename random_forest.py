import torch
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pdb

from utils import seed_everything
from datasets import LightCurveDataset
from models import *


def feature_extraction(dataset, model, device):
    x_train = torch.tensor(dataset.x_train, dtype=torch.float, device=device)
    x_val = torch.tensor(dataset.x_val, dtype=torch.float, device=device)
    seq_len_train = (x_train[:, :, 2] != 0).sum(dim=1)
    seq_len_val = (x_val[:, :, 2] != 0).sum(dim=1)
    model.eval()
    with torch.no_grad():
        _, train_features = model(x_train, seq_len_train)
        _, val_features = model(x_val, seq_len_val)
    train_features = train_features.cpu().numpy()
    val_features = val_features.cpu().numpy() 
    return train_features, val_features


seed_everything()

bs = 256
device = "cpu"
arch = "gru"
nin = 3
nh = 96
nl = 64
nout = 1
nlayers = 2
do = 0.25

dataset = LightCurveDataset("linear", fold=True, bs=bs, device=device, eval=True)
if arch == "gru":
    model = GRUAE(nin, nh, nl, nout, nlayers, do)
elif arch == "lstm":
    model = LSTMAE(nin, nh, nl, nout, nlayers, do)

model.to(device)
state_dict = torch.load("models/{}.pth".format(arch), map_location=device)
model.load_state_dict(state_dict)

train_features, val_features = feature_extraction(dataset, model, device)

train_features = np.concatenate((train_features, dataset.m_train[:, np.newaxis], dataset.s_train[:, np.newaxis], dataset.p_train[:, np.newaxis]), axis=1)
val_features = np.concatenate((val_features, dataset.m_val[:, np.newaxis], dataset.s_val[:, np.newaxis], dataset.p_val[:, np.newaxis]), axis=1)

y_train = dataset.y_train
y_val = dataset.y_val

skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(train_features, y_train)
print(skf)

# distributions = dict(C=uniform(loc=0, scale=4),
#                      penalty=['l2', 'l1'])

parameters = {
    "n_estimators": [50, 100, 250],
    "criterion": ["gini", "entropy"],
    "max_features": [3, 6, 12, 18],
    "min_samples_leaf": [1, 2, 3]
    }

rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters)

# val_idx are the validation index from k-cross validation 
for train_idx, val_idx in skf.split(train_features, y_train):
    clf.fit(train_features[train_idx], y_train[train_idx])
    val_score = clf.score(train_features[val_idx], y_train[val_idx])
    break