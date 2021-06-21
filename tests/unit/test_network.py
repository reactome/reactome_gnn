from collections import defaultdict
import networkx as nx
from reactome_gnn import network

graph = network.Network()


def test_graph():    
    assert isinstance(graph, network.Network)


def test_txt_type():
    assert isinstance(graph.graph_txt, defaultdict)


def test_txt_len():
    assert len(graph.graph_txt) > 0


def test_json_type():
    assert isinstance(graph.graph_dict, defaultdict)


def test_json():
    assert len(graph.graph_dict) > 0


def test_pathway_info_type():
    assert isinstance(graph.pathway_info, dict)


def test_pathway_info_len():
    assert len(graph.pathway_info) > 0


def test_parent_dict_type():
    assert isinstance(graph.parent_dict, defaultdict)


def test_parent_dict_type():
    assert len(graph.pathway_info) > 0


def test_pathway_parent_len():
    assert len(graph.pathway_info) == len(graph.parent_dict)


def test_name_id_type():
    assert isinstance(graph.name_to_id, dict)


def test_name_id_len():
    assert len(graph.name_to_id) <= len(graph.pathway_info)


def test_nx_graph():
    graph.to_networkx()
    assert isinstance(graph.graph_nx, nx.DiGraph)


def test_nx_num_nodes():
    assert len(graph.graph_nx.nodes) == len(graph.pathway_info)


def test_remove_by_id():
    id = 'R-HSA-109582'
    graph.remove_by_id(id)
    assert id not in graph.graph_dict


def test_remove_by_name():
    name = 'Signal Transduction'
    graph.remove_by_name(name)
    assert name not in graph.graph_dict


def test_id_child_not_present():
    # ok for now - but lets add these manually as I mentioned instead of testing them
    child_id = 'R-HSA-202733'
    assert child_id not in graph.graph_dict


def test_name_child_not_present():
    # ok for now - lets add these manually as I mentioned instead of testing them
    child_id = 'R-HSA-5663205'
    assert child_id not in graph.graph_dict
