import torch


def get_hyperparameters():
    """Returns a dictionary with the default hyperparameters."""
    return {
        'num_epochs': 200,
        'dim_latent': 16,
        'num_layers': 5,
        'batch_size': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 5e-4,
    }
