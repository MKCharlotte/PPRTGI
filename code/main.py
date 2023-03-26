import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import ndarray as nd
from scipy import interp
from sklearn import metrics
import warnings, sys
import networkx as nx
from train import Train
import argparse
import matplotlib
matplotlib.use('Agg')


parser = argparse.ArgumentParser()
parser.add_argument('--k', default='8')
parser.add_argument('--species')
parser.add_argument('--seqdim', type=int, default=256)
parser.add_argument('--embsize', type=int, default=256)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--aggregator', default='APPNPConv')
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--lr', type=float, default=0.0001)
args = parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_losses, val_losses = Train(dir="../data/",
                                      k=args.k,
                                      species=args.species,
                                      epochs=args.epochs,
                                      aggregator=args.aggregator,
                                      embedding_size=args.embsize,
                                      seq_hiddim=args.seqdim,
                                      layers=args.layers,
                                      dropout=0.2,
                                      slope=0.2,  # LeakyReLU
                                      alpha=args.alpha,
                                      lr=args.lr,
                                      wd=1e-3,
                                      random_seed=1,
                                      ctx=mx.gpu(0))
    iters = range(len(train_losses))
    plt.figure()
    # loss
    plt.plot(iters, train_losses, 'g', label='train loss')
    plt.plot(iters, val_losses, 'k', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.savefig('loss.jpg', dpi=800)
