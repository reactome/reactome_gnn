import os
import pickle

from reactome_gnn import marker, network


def create_network_from_markers(marker_list, p_value, study):
    """Perform enrichment analysis on the markers and create network.

    Given the list of markers and the p-value treshold, perform the
    enrichment analysis, and then create the network from the results
    of the enrichment analysis. Return that netowrk.

    Parameters
    ----------
    marker_list : list
        List of markers to use in enrichment analysis
    p_value : float
        Threshold for p-value to determine significant nodes
    study : str
        Name of the study

    Returns
    -------
    network.Network
        Network object created from the results of enrichment analysis
    """
    enrichment_analysis = marker.Marker(marker_list, p_value)
    graph = network.Network(enrichment_analysis.result, study)
    return graph


def create_network_from_stids(stid_list, study):
    """Create pathway network and highlight the specified stids."""
    graph = network.Network(study=study)
    graph.add_significance_by_stid(stid_list)
    return graph


def create_network_from_names(name_list, study):
    """Create pathway network and highlight the specified names."""
    graph = network.Network(study=study)
    graph.add_significance_by_name(name_list)
    return graph


def save_to_disk(graph, save_dir):
    """Save the network.Network object to disk."""
    assert os.path.isdir(save_dir), 'Directory does not exist!'
    save_path = os.path.join(save_dir, graph.study + '.pkl')
    pickle.dump(graph.graph_nx, open(save_path, 'wb'))


def create_toy_study_with_markers(p_value=0.05):
    """Create networks for the predefined set of markers.

    Given the four sets of markers (A, B, C, D), perform enrichment
    analysis for each set and create one network for each result.
    The four cases follow the patterns A = D, A & D subsets of C,
    B and C have no intersection. Saves the created networks to the
    disk and return them.

    Parameters
    ----------
    p_value : float
        Threshold for p-value to determine significant pathways

    Returns
    -------
    tuple
        four network.Network objects, one for each case in A, B, C, D
    """
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
    """Create networks for the predefined set of pathway names.

    Given the four sets of pathway names (A, B, C, D), create four
    "blank" networks and highlight only the specified names. The four
    cases follow the patterns A = D, A & D subsets of C, B and C have
    no intersection. Saves the created networks to the disk and return
    them.

    Returns
    -------
    tuple
        four network.Network objects, one for each case in A, B, C, D
    """
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