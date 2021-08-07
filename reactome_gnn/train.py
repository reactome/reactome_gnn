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
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader

import dataset
import model
import utils
import hyperparameters


def draw_loss_plot(train_loss, valid_loss, name):
    """Draw and save plot of train and validation loss over epochs.
    Parameters
    ----------
    train_loss : list
        List of training loss for each epoch
    valid_loss : list
        List of validation loss for each epoch
    name : str
        A name used for the file

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
    plt.savefig(f'loss_{name}.png')
    plt.show()


def train(hyperparams=None, data_path='demo/data/example'):
    """Train a network which will produce the embeddings.
    Parameters
    ----------
    hyperparams : dict, optional
        Dictionary with hyperparameters
    data_path : str, optional
        Relative path to where the data is stored

    Returns
    -------
    None
    """
    if hyperparams is None:
        hyperparams = hyperparameters.get_hyperparameters()

    # Unpack hyperparameters
    num_epochs = hyperparams['num_epochs']
    dim_latent = hyperparams['dim_latent']
    num_layers = hyperparams['num_layers']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']

    model_path = f'demo/data/model_dim{dim_latent}_lay{num_layers}.pth'
    
    # Create datasets
    ds = dataset.PathwayDataset(data_path)
    ds_train, ds_valid = [ds[0], ds[1], ds[2]], [ds[3], ds[4]]
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # Initialize networks and optimizer
    net = model.GCNModel(dim_latent, num_layers, train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_net = model.GCNModel(dim_latent, num_layers, train=True)
    best_net.load_state_dict(copy.deepcopy(net.state_dict()))

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
            

        running_loss = np.array(loss_per_graph).mean()
        loss_per_epoch_train.append(np.array(loss_per_graph).mean())
        print(f'Epoch: {epoch}\t\tTraining loss: {running_loss}')
        
        loss_per_graph = []
        net.eval()
        for data in dl_valid:
            graph, name = data
            name = name[0]
            logits = net(graph)
            labels = graph.ndata['significance'].unsqueeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss_per_graph.append(loss.item())
            
        running_loss = np.array(loss_per_graph).mean()
        if len(loss_per_epoch_valid) > 0 and running_loss < min(loss_per_epoch_valid):
            best_net.load_state_dict(copy.deepcopy(net.state_dict()))
        loss_per_epoch_valid.append(running_loss)
        print(f'Epoch: {epoch}\t\tValidation loss: {running_loss}')

    draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, 'plot')

    torch.save(best_net.state_dict(), model_path)


if __name__ == '__main__':
    train()
