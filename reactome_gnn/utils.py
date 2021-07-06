import os
import pickle

from networkx.generators.atlas import graph_atlas

from reactome_gnn import marker, network


def create_network_from_markers(marker_list, p_value, study):
    enrichment_analysis = marker.Marker(marker_list, p_value)
    graph = network.Network(enrichment_analysis.result, study)
    return graph


def create_network_from_stids(stid_list, study):
    graph = network.Network(study=study)
    graph.add_significance_by_stid(stid_list)
    return graph


def create_network_from_names(name_list, study):
    graph = network.Network(study=study)
    graph.add_significance_by_name(name_list)
    return graph


def save_to_disk(graph, save_dir):
    assert os.path.isdir(save_dir), 'Directory does not exist!'
    save_path = os.path.join(save_dir, graph.study + '.pkl')
    pickle.dump(graph.graph_nx, open(save_path, 'wb'))


def create_toy_study_with_markers(p_value=0.05):
    study_A = ['RAS', 'MAP', 'STAT']
    study_B = ['EGF', 'EGFR']
    study_C = ['RAS', 'MAP', 'IL10', 'STAT']
    study_D = ['RAS', 'MAP', 'STAT']
    
    graph_A = create_network_from_markers(study_A, p_value, 'study_A')
    graph_B = create_network_from_markers(study_B, p_value, 'study_B')
    graph_C = create_network_from_markers(study_C, p_value, 'study_C')
    graph_D = create_network_from_markers(study_D, p_value, 'study_D')

    save_dir = 'data/example'
    save_to_disk(graph_A, save_dir)
    save_to_disk(graph_B, save_dir)
    save_to_disk(graph_C, save_dir)
    save_to_disk(graph_D, save_dir)

    return graph_A, graph_B, graph_C, graph_D


def create_toy_study_with_names():
    study_A = ["Signaling by WNT", "WNT ligand biogenesis and trafficking",
               "Degradation of beta-catenin by the destruction complex",
               "TCF dependent signaling in response to WNT",
               "Beta-catenin independent WNT signaling"]
    study_B = ["Autophagy", "Macroautophagy", "Chaperone Mediated Autophagy",
               "Late endosomal microautophagy"]
    study_C = ["Signal Transduction", "Signaling by NOTCH", "Signaling by NOTCH1",
               "Signaling by NOTCH2", "Signaling by NOTCH3", "Signaling by NOTCH4",
               "Activated NOTCH1 Transmits Signal to the Nucleus",
               "NOTCH1 Intracellular Domain Regulates Transcription",
               "Signaling by WNT", "WNT ligand biogenesis and trafficking",
               "Degradation of beta-catenin by the destruction complex",
               "TCF dependent signaling in response to WNT",
               "Beta-catenin independent WNT signaling"]
    study_D = ["Signaling by WNT", "WNT ligand biogenesis and trafficking",
               "Degradation of beta-catenin by the destruction complex",
               "TCF dependent signaling in response to WNT",
               "Beta-catenin independent WNT signaling"]
    
    graph_A = create_network_from_names(study_A, 'study_A')
    graph_B = create_network_from_names(study_B, 'study_B')
    graph_C = create_network_from_names(study_C, 'study_C')
    graph_D = create_network_from_names(study_D, 'study_D')

    save_dir = 'data/example'
    save_to_disk(graph_A, save_dir)
    save_to_disk(graph_B, save_dir)
    save_to_disk(graph_C, save_dir)
    save_to_disk(graph_D, save_dir)

    return graph_A, graph_B, graph_C, graph_D