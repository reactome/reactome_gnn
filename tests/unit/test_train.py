import os
from reactome_gnn import train

def test_train():
    hyperparams = {
        'num_epochs' : 2,
        'dim_latent': 2,
        'num_layers': 2,
        'batch_size': 1,
        'device': 'cpu',
        'lr': 1e-3,
    }
    model_path = train.train(hyperparams=hyperparams, plot=False)
    assert os.path.isfile(model_path)
