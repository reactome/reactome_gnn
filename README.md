# Reactome_GNN


[![Build Status](https://app.travis-ci.com/reactome/reactome_gnn.svg?branch=main)](travis-ci.com/reactome/reactome_gnn)

Reactome_GNN is a Python package for creating higly-dimensional representations of human pathway networks, which can then be used in downstream tasks.

The current version uses a multi-layer Graph Convolutional Network (GCN) as a model for obtaining these representations.



## Installation
```bash
git clone https://github.com/reactome/reactome_gnn.git
cd reactome_gnn
python setup.py install
```

### Dependencies
- Python 3.8
- Pip 20.2.4+

## Usage
You can find a jupyter notebook with usage instructions in the `demo` directory. There you can find a step-by-step guide on how to generate networks in several different ways, transform them into DGL graphs, specify the GNN model, and obtain the embeddings.
