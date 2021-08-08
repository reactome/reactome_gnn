import torch


def get_hyperparameters():
    """Returns a dictionary with the default hyperparameters."""
    return {
        'num_epochs': 100,
        'dim_latent': 8,
        'num_layers': 1,
        'batch_size': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-3,
    }
