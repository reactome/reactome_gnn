import argparse
from datetime import datetime
import copy
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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
    plt.savefig(f'figures/loss_{timestamp}.png')
    plt.show()

def train():
    path = ''
    # model_path = os.path.abspath(os.path.join(data_dir, 'models/model.pth'))
    net = model.GCNModel(8, 1)
    ds = dataset.PathwayDataset('demo/data/example')

    pass

