"""
A deep MNIST classifier using convolutional layers.

This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import os
import argparse
import logging
import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nni.utils import merge_parameter
from torchvision import datasets, transforms
from collections.abc import Iterable

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import CategoricalImputer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
from torchvision import models
from torch.nn import init
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.optim import lr_scheduler

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pandas as pd
import numpy as np
import sys

logger = logging.getLogger('AutoDL')


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0., actn=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


def ifnone(a, b):
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def listify(p, q):
    "Make `p` listy and the same length as `q`."
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    # Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try:
            a = len(p)
        except:
            p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1: p = p * n
    assert len(p) == n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, device, ps=None,
                 emb_drop=0., y_range=None, use_bn: bool = True, bn_final: bool = False):
        super().__init__()
        ps = ifnone(ps, [0] * len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList(
            [nn.Embedding(ni, nf) for ni, nf in emb_szs])  # type: torch.nn.modules.container.ModuleList
        self.emb_drop = nn.Dropout(emb_drop)  # type: torch.nn.modules.dropout.Dropout
        self.bn_cont = nn.BatchNorm1d(n_cont)  # type torch.nn.modules.batchnorm.BatchNorm1d
        n_emb = sum(e.embedding_dim for e in self.embeds)  # n_emb = 17 , type: int
        self.n_emb, self.n_cont, self.y_range = n_emb, n_cont, y_range
        sizes = [n_emb + n_cont] + layers + [out_sz]  # typeL list, len: 4
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes) - 2)] + [
            None]  # type: list, len: 3.  the last in None because we finish with linear
        layers = []
        for i, (n_in, n_out, dp, act) in enumerate(zip(sizes[:-1], sizes[1:], [0.] + ps, actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i != 0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)  # type: torch.nn.modules.container.Sequential
        self.device = device

    def forward(self, x_cat, x_cont):

        if self.n_emb != 0 and x_cat is not None:
            x_cat = x_cat.to(self.device)
            x = [e(x_cat[:, i]) for i, e in enumerate(
                self.embeds)]  # take the embedding list and grab an embedding and pass in our single row of data.
            x = torch.cat(x, 1)  # concatenate it on dim 1 ## remeber that the len is the batch size
            x = self.emb_drop(x)  # pass it through a dropout layer

        if self.n_cont != 0 and x_cont is not None:
            x_cont = x_cont.to(self.device)
            x_cont = self.bn_cont(x_cont)  # batchnorm1d
            x = torch.cat([x, x_cont],
                          1) if self.n_emb != 0 else x_cont  # combine the categircal and continous variables on dim 1
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1] - self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]  # deal with y_range
        x = x.squeeze()
        x = torch.sigmoid(x)
        return x


class ColumnarDataset(torch.utils.data.Dataset):
    def __init__(self, df, cats, y):
        self.dfcats = df[cats]  # type: pandas.core.frame.DataFrame
        self.dfconts = df.drop(cats, axis=1)  # type: pandas.core.frame.DataFrame

        if self.dfcats.shape[1] > 0:
            self.cats = np.stack([c.values for n, c in self.dfcats.items()], axis=1).astype(
                np.int64)  # tpye: numpy.ndarray
        if self.dfconts.shape[1] > 0:
            self.conts = np.stack([c.values for n, c in self.dfconts.items()], axis=1).astype(
                np.float32)  # tpye: numpy.ndarray
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.dfcats.shape[1] > 0 and self.dfconts.shape[1] > 0:
            return [self.cats[idx], self.conts[idx], self.y[idx]]
        if self.dfcats.shape[1] <= 0 and self.dfconts.shape[1] > 0:
            return [0, self.conts[idx], self.y[idx]]
        if self.dfcats.shape[1] > 0 and self.dfconts.shape[1] <= 0:
            return [self.cats[idx], 0., self.y[idx]]
        return [0, 0., self.y[idx]]


class BCE_loss_weighted(nn.Module):
    def __init__(self):
        super(BCE_loss_weighted, self).__init__()

    def forward(self, pred, y):
        # pred is the model predicted probabilities
        # y is labels

        n_class1 = (y == 1).sum().to(dtype=torch.float)
        n_class0 = (y == 0).sum().to(dtype=torch.float)

        pred = pred.to(dtype=torch.float).squeeze()
        y = y.to(dtype=torch.float)

        weight = torch.tensor([max(n_class1, n_class0) / n_class0, max(n_class1, n_class0) / n_class1])
        weight_ = weight[y.data.view(-1).long()].view_as(y)

        loss_fn = nn.BCELoss(reduce=False)
        loss = (loss_fn(pred, y) * weight_).mean()

        return loss


def weighted_binary_acc(y_pred, y_test):
    y_pred_tag = np.array([y >= 0.5 for y in y_pred]).astype(float)
    y_test = np.array(y_test)
    TP = ((y_pred_tag == y_test) * (y_test == 1)).sum()
    TN = ((y_pred_tag == y_test) * (y_test == 0)).sum()
    N = (y_test == 0).sum()
    P = (y_test == 1).sum()

    return 0.5 * (TP / P + TN / N)


def binary_acc(y_pred, y_test):
    y_pred_tag = np.array([y >= 0.5 for y in y_pred]).astype(float)
    y_test = np.array(y_test)
    correct_results_sum = (y_pred_tag == y_test).sum()
    acc = correct_results_sum / len(y_test)

    return acc


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (cat, cont, target) in enumerate(train_loader):
        cat, cont, target = cat.to(device), cont.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(cat, cont)
        loss = BCE_loss_weighted()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(cont), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, metric):
    model.eval()
    test_loss = 0
    y_true_val = list()
    y_pred_val = list()
    with torch.no_grad():
        for cat, cont, target in test_loader:
            cat, cont, target = cat.to(device), cont.to(device), target.to(device)
            output = model(cat, cont)
            # sum up batch loss
            test_loss += BCE_loss_weighted()(output, target)
            # get the index of the max log-probability
            y_true_val += list(target.cpu().data.numpy())
            y_pred_val += list(output.cpu().data.numpy())

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * metric(y_pred_val, y_true_val)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
        test_loss, accuracy))

    return accuracy


def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    torch.manual_seed(args['seed'])
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    path_train = sys.argv[1]

    df = pd.read_csv(path_train)

    target = sys.argv[2]

    X = df.drop(target, 1)
    y = df[target]

    num_attribs = list(X.select_dtypes(include=[np.number]))
    cat_attribs = list(X.columns.drop(num_attribs))


    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),  # replace Null by median
        ('std_scaler', StandardScaler()),  # mean 0 std 1
    ])

    cat_pipeline = Pipeline([
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

    # Training Data prepared
    X_prepared = full_pipeline.fit_transform(X)
    ordinal_encoder = OrdinalEncoder()
    y_prepared = [x[0] for x in ordinal_encoder.fit_transform(pd.DataFrame(y))]

    # Splitting to train and test
    X_train, X_val, y_train, y_val = train_test_split(X_prepared, y_prepared, test_size=0.2, random_state=42)

    for col in cat_attribs:
        df[col] = df[col].astype('category')

    cat_szs = [len(df[col].cat.categories) for col in cat_attribs]
    emb_szs = [(size, min(50, (size + 1) // 2)) for size in cat_szs]

    X_train_df = pd.DataFrame(X_train, columns=num_attribs + cat_attribs)
    X_val_df = pd.DataFrame(X_val, columns=num_attribs + cat_attribs)

    trainds = ColumnarDataset(X_train_df, cat_attribs, np.array(y_train))
    valds = ColumnarDataset(X_val_df, cat_attribs, np.array(y_val))

    params = {'batch_size': 2048,
              'shuffle': True}

    traindl = torch.utils.data.DataLoader(trainds, **params)
    valdl = torch.utils.data.DataLoader(valds, **params)

    hidden_size = args['hidden_size']
    model = TabularModel(emb_szs=emb_szs, n_cont=len(num_attribs), out_sz=1, layers=[hidden_size, hidden_size // 2],
                         ps=[args['ps1'], args['ps2']], emb_drop=args['emb_drop'], device=device).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, traindl, optimizer, epoch)
        test_acc = test(args, model, device, valdl, weighted_binary_acc)

        print(test_acc)
        # report intermediate result
        nni.report_intermediate_result(test_acc)
        logger.debug('test accuracy %g', test_acc)
        logger.debug('Pipe send intermediate result done.')

    # report final result
    nni.report_final_result(test_acc)
    logger.debug('Final result is %g', test_acc)
    logger.debug('Send final result done.')


def get_params():
    # Training settings

    parser = argparse.ArgumentParser(description='AutoDL')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--ps1', type=float, default=0.1, help='Dropout for first layer')
    parser.add_argument('--ps2', type=float, default=0.05, help='Dropout for second layer')
    parser.add_argument('--emb_drop', type=float, default=0.04, help='Dropout for categorical embeddings')


    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
