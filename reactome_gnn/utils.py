import os
import pickle

import torch

from reactome_gnn import marker, network, dataset, model


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


def create_embeddings(dim_latent=1, num_layers=1, load_model=True):
    """Create embeddings for all the graphs stored on the disk.

    First the Pathway dataset is created which takes all the graphs
    stored in the 'raw' directory, processes them, and stores them in
    the 'processed' directory. Each graph is fed to the model with
    specified latent dimension and number of GCN layers, which returns
    the embedding of that graph. All the embeddings are saved on the
    disk in the 'embeddings' directory.

    Parameters
    ----------
    dim_latent : int, optional
        Dimension of the graph embeddings, default 16
    num_layers : int, optional
        Number of GCN layers in the GCNModel, deafult 1
    load_model : bool, optional
        Whether to load an old model or create a new one
    """
    data = dataset.PathwayDataset('data/example')
    emb_dir = os.path.abspath('data/example/embeddings')
    model_path = os.path.abspath('data/example/models/model.pth')
    if not os.path.isdir(emb_dir):
        os.mkdir(emb_dir)
    net = model.GCNModel(dim_latent=dim_latent, num_layers=num_layers)
    if load_model:
        net.load_state_dict(torch.load(model_path))
    else:
        torch.save(net.state_dict(), model_path)
    for idx in range(len(data)):
        graph, name = data[idx]
        embedding = net(graph).detach()
        emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
        torch.save(embedding, emb_path)


def get_embedding(name):
    """Load and return the embedding of the graph with specified index."""
    emb_path = os.path.abspath(f'data/example/embeddings')
    embedding = torch.load(os.path.join(emb_path, f'{name}.pth'))
    return embedding


def embeddings_sanity_check():
    """Check whether the embeddings of the toy examples are consistent between themselves."""
    studies = []
    for c in 'ABCD':
        studies.append(get_embedding(f'study_{c}'))
    stids = pickle.load(open('data/example/info/sorted_stid_list.pkl', 'rb'))
    name_to_id = pickle.load(open('data/example/info/name_to_id.pkl', 'rb'))
    test_1 = [studies[i][stids.index(name_to_id["Late endosomal microautophagy"])].item() for i in range(4)]
    assert test_1[0] == test_1[2] == test_1[3] != test_1[1]
    test_2 = [studies[i][stids.index(name_to_id["Signaling by NOTCH"])].item() for i in range(4)]
    assert test_2[0] == test_2[1] == test_2[3] != test_1[2]
    test_3 = [studies[i][stids.index(name_to_id["Macroautophagy"])].item() for i in range(4)]
    assert test_3[0] == test_3[2] == test_3[3] != test_3[1]
    assert (studies[0] == studies[3]).all()
    print("Passed all the tests!")
    return True