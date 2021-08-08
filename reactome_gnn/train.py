import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    ds_train, ds_valid = [ds[0], ds[1], ds[2]], [ds[3], ds[4]]
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    # Initialize networks and optimizer
    net = model.GCNModel(dim_latent, num_layers, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_net = model.GCNModel(dim_latent, num_layers, do_train=True)
    best_net.load_state_dict(copy.deepcopy(net.state_dict()))

    loss_per_epoch_train, loss_per_epoch_valid = [], []

    # Start training
    for epoch in range(num_epochs):

        # Training iteration
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

            loss_per_graph.append(loss.item())
            
        # Output loss
        running_loss = np.array(loss_per_graph).mean()
        loss_per_epoch_train.append(np.array(loss_per_graph).mean())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}\t\tTraining loss: {running_loss}')
        
        # Validation iteration
        with torch.no_grad():
            loss_per_graph = []
            net.eval()
            for data in dl_valid:
                graph, name = data
                name = name[0]
                logits = net(graph)
                labels = graph.ndata['significance'].unsqueeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss_per_graph.append(loss.item())
            
        # Output loss
        running_loss = np.array(loss_per_graph).mean()
        if len(loss_per_epoch_valid) > 0 and running_loss < min(loss_per_epoch_valid):
            best_net.load_state_dict(copy.deepcopy(net.state_dict()))
        loss_per_epoch_valid.append(running_loss)

        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch+1}\t\tValidation loss: {running_loss}')

    # Plot loss
    if plot:
        plot_path = os.path.join(data_path, 'figures')
        plot_path = os.path.join(plot_path, f'plot_dim{dim_latent}_lay{num_layers}.png')
        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, plot_path)

    # Save the best model
    torch.save(best_net.state_dict(), model_path)

    return model_path
