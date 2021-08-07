import argparse
from datetime import datetime
import copy
import os
import pickle
import random
import time
from dgl.batch import _batch_feat_dicts, batch

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader

import dataset
import model
import utils


def draw_loss_plot(train_loss, valid_loss, timestamp):
    """Draw and save plot of train and validation loss over epochs.
    Parameters
    ----------
    train_loss : list
        List of training loss for each epoch
    valid_loss : list
        List of validation loss for each epoch
    timestamp : str
        A timestep used for naming the file
    Returns
    -------
    None
    """
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_{timestamp}.png')
    plt.show()

def train():
    path = ''
    dim_latent = 8
    num_layers = 1
    num_epochs = 100
    learning_rate = 1e-3
    device = 'cpu'
    # model_path = os.path.abspath(os.path.join(data_dir, 'models/model.pth'))
    net = model.GCNModel(dim_latent, num_layers, train=True)
    ds = dataset.PathwayDataset('demo/data/example')
    ds_train, ds_valid = [ds[0], ds[1], ds[2]], [ds[3]]
    dl_train = GraphDataLoader(ds_train, batch_size=1, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=1, shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    loss_per_epoch_train, loss_per_epoch_valid = [], []

    for epoch in range(num_epochs):
        loss_per_graph = []   
        net.train() 
        for data in dl_train:
            graph, name = data
            name = name[0]
            logits = net(graph)
            labels = graph.ndata['significance'].unsqueeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Plot loss
            loss_per_graph.append(loss.item())

        loss_per_epoch_train.append(np.array(loss_per_graph).mean())
        
        loss_per_graph = []
        net.eval()
        for data in dl_valid:
            graph, name = data
            name = name[0]
            logits = net(graph)
            labels = graph.ndata['significance'].unsqueeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss_per_graph.append(loss.item())

        loss_per_epoch_valid.append(np.array(loss_per_graph).mean())

    draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, 'plot')

if __name__ == '__main__':
    train()