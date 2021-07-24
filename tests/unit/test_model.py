import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv
from reactome_gnn import model


net = model.GCNModel(dim_latent=8, num_layers=3)
graph = dgl.load_graphs('data/example/processed/study_A.dgl')[0][0]
embedding = net(graph)


def test_embedder():
    assert isinstance(net.embedder, nn.Module)


def test_linear():
    assert isinstance(net.linear, nn.Module)


def test_relu():
    assert isinstance(net.relu, nn.Module)


def test_conv_0():
    assert isinstance(net.conv_0, GraphConv)


def test_conv_layers_type():
    assert isinstance(net.layers, nn.ModuleList)


def test_conv_layers_len():
    assert len(net.layers) == 2


def test_embedding_type():
    assert isinstance(embedding, torch.Tensor)


def test_embedding_num():
    assert embedding.shape[0] == graph.num_nodes()


def test_embedding_dim():
    assert embedding.shape[1] == 8
