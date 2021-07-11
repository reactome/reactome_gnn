import torch
import torch.nn as nn
from dgl.nn import GraphConv


class GCNModel(nn.Module):
    def __init__(self, dim_latent, num_layers=1):
        super().__init__()
        self.embedder = nn.Embedding(2, 2)
        self.conv_0 = GraphConv(3, dim_latent, allow_zero_in_degree=True)
        self.relu = nn.LeakyReLU()
        self.layers = nn.ModuleList([GraphConv(dim_latent, dim_latent, allow_zero_in_degree=True)
                                     for _ in range(num_layers - 1)])

    def forward(self, graph):
        significance = graph.ndata['significance'].int()
        significance = self.embedder(significance)
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = torch.cat((weights, significance), dim=1)
        embedding = self.conv_0(graph, features)
        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)
        return embedding
