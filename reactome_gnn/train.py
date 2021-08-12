import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import GraphDataLoader

from reactome_gnn import model, dataset, hyperparameters


def draw_loss_plot(train_loss, valid_loss, save_path):
    """Draw and save plot of train and validation loss over epochs.

    Parameters
    ----------
    train_loss : list
        List of training loss for each epoch
    valid_loss : list
        List of validation loss for each epoch
    save_path : str
        A path where the plot will be saved

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
    plt.savefig(f'{save_path}.png')
    plt.show()


def draw_f1_plot(train_f1, valid_f1, save_path):
    """Draw and save plot of F1 score calculated during validation.

    Parameters
    ----------
    f1 scores : list
        List of f1 scores during validation
    save_path : str
        A path where the plot will be saved

    Returns
    -------
    None
    """
    plt.figure()
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='validation')
    plt.title('F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(f'{save_path}.png')
    plt.show()


def train(hyperparams=None, data_path='demo/data/example', plot=True):
    """Train a network which will produce the embeddings.

    Parameters
    ----------
    hyperparams : dict, optional
        Dictionary with hyperparameters
    data_path : str, optional
        Relative path to where the data is stored
    plot : bool, optional
        Whether to plot loss over epochs

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

    model_path = os.path.join(data_path, 'models')
    model_path = os.path.join(model_path, f'model_dim{dim_latent}_lay{num_layers}.pth')
    
    # Create datasets
    ds = dataset.PathwayDataset(data_path)
    ds_train = [ds[0], ds[1], ds[2], ds[5], ds[6], ds[7], ds[8], ds[9]]
    ds_valid = [ds[3], ds[4], ds[10], ds[11]]
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # Initialize networks and optimizer
    net = model.GCNModel(dim_latent, num_layers, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_net = model.GCNModel(dim_latent, num_layers, do_train=True)
    best_net.load_state_dict(copy.deepcopy(net.state_dict()))

    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_per_epoch_train, f1_per_epoch_valid = [], []

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    weight = torch.tensor([0.00001, 0.99999]).to(device)

    # Start training
    for epoch in range(num_epochs):

        # Training iteration
        loss_per_graph = []
        f1_per_graph = [] 
        net.train()
        for data in dl_train:
            graph, name = data
            name = name[0]
            logits = net(graph)
            labels = graph.ndata['significance'].unsqueeze(-1)
            weight_ = weight[labels.data.view(-1).long()].view_as(labels)

            # Get weighted loss
            loss = criterion(logits, labels)
            loss_weighted = loss * weight_
            loss_weighted = loss_weighted.mean()

            # Update parameters
            optimizer.zero_grad()
            loss_weighted.backward()
            optimizer.step()

            # Append output metrics
            loss_per_graph.append(loss_weighted.item())
            preds = logits.sigmoid().squeeze(1).int()
            labels = labels.squeeze(1).int()
            f1 = metrics.f1_score(labels, preds)
            f1_per_graph.append(f1)
            
        # Output loss and f1
        running_loss = np.array(loss_per_graph).mean()
        running_f1 = np.array(f1_per_graph).mean()
        loss_per_epoch_train.append(running_loss)
        f1_per_epoch_train.append(running_f1)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch: {epoch+1}\tTraining loss:\t\t{running_loss}')
        
        # Validation iteration
        with torch.no_grad():
            loss_per_graph = []
            f1_per_graph = []
            net.eval()
            for data in dl_valid:
                graph, name = data
                name = name[0]
                logits = net(graph)
                labels = graph.ndata['significance'].unsqueeze(-1)

                # Get weighted loss
                weight_ = weight[labels.data.view(-1).long()].view_as(labels)
                loss = criterion(logits, labels)
                loss_weighted = loss * weight_
                loss_weighted = loss_weighted.mean()

                # Append output metrics
                loss_per_graph.append(loss_weighted.item())
                preds = logits.sigmoid().squeeze(1).int()
                labels = labels.squeeze(1).int()
                f1 = metrics.f1_score(labels, preds)
                f1_per_graph.append(f1)
            
        # Output loss and f1
        running_loss = np.array(loss_per_graph).mean()
        running_f1 = np.array(f1_per_graph).mean()
        loss_per_epoch_valid.append(running_loss)
        f1_per_epoch_valid.append(running_f1)

        # Save the best model
        if len(loss_per_epoch_valid) > 0 and running_loss < min(loss_per_epoch_valid):
            best_net.load_state_dict(copy.deepcopy(net.state_dict()))

        if (epoch+1) % 20 == 0:
            print(f'Epoch: {epoch+1}\tValidation loss:\t{running_loss}')
            print(f'Epoch: {epoch+1}\tF1 score:\t\t{running_f1}\n')

    # Plot loss
    if plot:
        plot_path = os.path.join(data_path, 'figures')
        loss_path = os.path.join(plot_path, f'loss_dim{dim_latent}_lay{num_layers}.png')
        f1_path = os.path.join(plot_path, f'f1_dim{dim_latent}_lay{num_layers}.png')
        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

    # Save the best model
    torch.save(best_net.state_dict(), model_path)

    return model_path

if __name__ == '__main__':
    train()