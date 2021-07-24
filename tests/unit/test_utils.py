import torch
from sklearn.cross_decomposition import CCA
from reactome_gnn import utils, network


def test_create_network_from_markers():
    marker_list = ['RAS', 'MAP', 'IL10', 'EGF', 'EGFR', 'STAT']
    p_value = 0.05
    graph = utils.create_network_from_markers(marker_list, p_value, 'test')
    assert isinstance(graph, network.Network)


def test_create_network_from_stids():
    stids = ['RAS', 'MAP', 'IL10', 'EGF', 'EGFR', 'STAT']
    graph = utils.create_network_from_stids(stids, 'test')
    assert isinstance(graph, network.Network)


def test_create_network_from_names():
    names = ["Autophagy", "Macroautophagy", "Chaperone Mediated Autophagy",
             "Late endosomal microautophagy"]
    graph = utils.create_network_from_names(names, 'test')
    assert isinstance(graph, network.Network)


def test_get_embedding():
    emb = utils.get_embedding('study_A')
    assert isinstance(emb, torch.Tensor)


def test_fit_cca_on_toy_data():
    cca = utils.fit_cca_on_toy_data()
    assert isinstance(cca, CCA)
