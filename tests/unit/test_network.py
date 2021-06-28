from collections import defaultdict
import networkx as nx
from networkx.generators.atlas import graph_atlas
from reactome_gnn import marker, network

ea_result = marker.Marker(marker_list=['EGF', 'EGFR'], p_value=0.05)
graph = network.Network(ea_result=ea_result.result, study='Test')


def test_graph():    
    assert isinstance(graph, network.Network)


def test_txt_type():
    assert isinstance(graph.txt_adjacency, dict)


def test_txt_len():
    assert len(graph.txt_adjacency) > 0


def test_json_type():
    assert isinstance(graph.json_adjacency, dict)


def test_json():
    assert len(graph.json_adjacency) > 0


def test_pathway_info_type():
    assert isinstance(graph.pathway_info, dict)


def test_pathway_info_len():
    assert len(graph.pathway_info) > 0


def test_name_id_type():
    assert isinstance(graph.name_to_id, dict)


def test_name_id_len():
    # Should be equal, but some pathways have the same name but different stId 
    assert len(graph.name_to_id) <= len(graph.pathway_info)


def test_nx_graph():
    graph.to_networkx()
    assert isinstance(graph.graph_nx, nx.DiGraph)


def test_nx_num_nodes():
    assert len(graph.graph_nx.nodes) == len(graph.pathway_info)


def test_remove_by_id():
    # R-HSA-1266738 is the stId of 'Developmental Biology' pathway
    id = 'R-HSA-1266738'
    graph.remove_by_id(id)
    assert id not in graph.graph_nx


def test_remove_by_name():
    name = 'Signal Transduction'
    graph.remove_by_name(name)
    stid = graph.name_to_id[name]
    assert stid not in graph.graph_nx


def test_id_child_not_present():
    # This is for testing whether the children od the nodes specified above (remove_by_id) were removed correctly
    # R-HSA-186712 is the child node of the 'Developmental Biology' node
    child_id = 'R-HSA-186712'
    assert child_id not in graph.graph_nx


def test_name_child_not_present():
    # This is for testing whether the children od the nodes specified above (remove_by_name) were removed correctly
    # R-HSA-9716542 is a child node of the 'Signal Transduction' node
    child_id = 'R-HSA-9716542'
    assert child_id not in graph.graph_nx


def test_nx_attribute_stid():
    stid = 'R-HSA-556833'  # stId for the 'Metabolism of lipids' pathway
    assert graph.graph_nx.nodes[stid]['stId'] == 'R-HSA-556833'


def test_nx_attribute_name():
    stid = 'R-HSA-556833'  # stId for the 'Metabolism of lipids' pathway
    assert graph.graph_nx.nodes[stid]['name'] == 'Metabolism of lipids'


def test_nx_attribute_weight():
    stid = 'R-HSA-556833'  # stId for the 'Metabolism of lipids' pathway
    assert graph.graph_nx.nodes[stid]['weight'] == graph.weights[stid]['p_value']


def test_nx_attribute_significance():
    stid = 'R-HSA-556833'  # stId for the 'Metabolism of lipids' pathway
    assert graph.graph_nx.nodes[stid]['significance'] == graph.weights[stid]['significance']
