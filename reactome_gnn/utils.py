import os
import pickle

import dgl
import torch
from sklearn.cross_decomposition import CCA

from reactome_gnn import marker, network, dataset, model, train, hyperparameters


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


def create_toy_study_with_markers(p_value=0.05, save=False,
                                  data_dir='demo/data/example'):
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
    save : bool, optional
        Whether to save the created networks to the disk, default: False
    data_dir : str, optional
        Relative path to where the data is stored

    Returns
    -------
    tuple
        four network.Network objects, one for each case in A, B, C, D
    """
    study_A = ['RAS', 'MAP', 'STAT']
    study_B = ['EGF', 'EGFR']
    study_C = ['RAS', 'MAP', 'IL10', 'STAT']
    study_D = ['RAS', 'MAP', 'STAT']
    study_E = ['ABCB4', 'ECHS1', 'GLRX5', 'KCNQ2', 'PEX14']
    study_F = ['SHROOM1', 'SMARCA2', 'TMEM62', 'CYP19A1']
    study_G = ['BCR', 'CCL16', 'CPVL', 'GAA']
    study_H = ['HMGCS2', 'ITIH', 'LAMB2', 'MAPK13']
    study_I = ['NDUFA4', 'POR', 'RNF', 'SERPINA1']
    study_J = ['SLC', 'TCHP', 'TIMM13', 'UBR4']
    study_K = ['COL6A3', 'FH', 'GRHPR', 'IL1R1']
    study_L = ['ITIH3', 'MAT1A', 'NUDT2', 'PPARGC1B', 'SCN1A']

    graph_A = create_network_from_markers(study_A, p_value, 'study_A')
    graph_B = create_network_from_markers(study_B, p_value, 'study_B')
    graph_C = create_network_from_markers(study_C, p_value, 'study_C')
    graph_D = create_network_from_markers(study_D, p_value, 'study_D')
    graph_E = create_network_from_markers(study_E, p_value, 'study_E')
    graph_F = create_network_from_markers(study_F, p_value, 'study_F')
    graph_G = create_network_from_markers(study_G, p_value, 'study_G')
    graph_H = create_network_from_markers(study_H, p_value, 'study_H')
    graph_I = create_network_from_markers(study_I, p_value, 'study_I')
    graph_J = create_network_from_markers(study_J, p_value, 'study_J')
    graph_K = create_network_from_markers(study_K, p_value, 'study_K')
    graph_L = create_network_from_markers(study_L, p_value, 'study_L')

    if save:
        save_dir = os.path.join(data_dir, 'raw')
        save_to_disk(graph_A, save_dir)
        save_to_disk(graph_B, save_dir)
        save_to_disk(graph_C, save_dir)
        save_to_disk(graph_D, save_dir)
        save_to_disk(graph_E, save_dir)
        save_to_disk(graph_F, save_dir)
        save_to_disk(graph_G, save_dir)
        save_to_disk(graph_H, save_dir)
        save_to_disk(graph_I, save_dir)
        save_to_disk(graph_J, save_dir)
        save_to_disk(graph_K, save_dir)
        save_to_disk(graph_L, save_dir)


    return graph_A, graph_B, graph_C, graph_D, graph_E, graph_F, graph_G, graph_H, graph_I, graph_J, graph_K, graph_L


def create_toy_study_with_names(save=False, data_dir='demo/data/example'):
    """Create networks for the predefined set of pathway names.

    Given the four sets of pathway names (A, B, C, D), create four
    "blank" networks and highlight only the specified names. The four
    cases follow the patterns A = D, A & D subsets of C, B and C have
    no intersection. Saves the created networks to the disk and return
    them.

    Parameters
    ----------
    save : bool, optional
        Whether to save the created networks to the disk, default: False
    data_dir : str, optional
        Relative path to where the data is stored

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
    study_E = ["DNA Repair", "DNA Damage Bypass",
               "Recognition of DNA damage by PCNA-containing replication complex",
               "Translesion synthesis by Y family DNA polymerases bypasses lesions on DNA template",
               "Translesion synthesis by REV1", "Translesion synthesis by POLK"]

    graph_A = create_network_from_names(study_A, 'study_A')
    graph_B = create_network_from_names(study_B, 'study_B')
    graph_C = create_network_from_names(study_C, 'study_C')
    graph_D = create_network_from_names(study_D, 'study_D')
    graph_E = create_network_from_names(study_E, 'study_E')

    if save:
        save_dir = os.path.join(data_dir, 'raw')
        save_to_disk(graph_A, save_dir)
        save_to_disk(graph_B, save_dir)
        save_to_disk(graph_C, save_dir)
        save_to_disk(graph_D, save_dir)
        save_to_disk(graph_E, save_dir)

    return graph_A, graph_B, graph_C, graph_D, graph_E


def create_embeddings(load_model=True, save=False, data_dir='demo/data/example',
                      hyperparams=None, plot=True):
    """Create embeddings for all the graphs stored on the disk.

    First the Pathway dataset is created which takes all the graphs
    stored in the 'raw' directory, processes them, and stores them in
    the 'processed' directory. Each graph is fed to the model with
    specified latent dimension and number of GCN layers, which returns
    the embedding of that graph. All the embeddings are saved on the
    disk in the 'embeddings' directory.

    Parameters
    ----------
    load_model : bool, optional
        Whether to load an old model or create a new one
    save : bool, optional
        Whether to save the created embeddings to the disk, default: False
    data_dir : str, optional
        Relative path to where the data is stored
    hyperparams : dict, optional
        In case no hyperparameter dictionary is passed, default
        hyperparameters are used
    plot : bool, optional
        In case of training the model, whether loss should be plotted

    Returns
    -------
    list
        A list of embedding-tuples, the first element in the tuple is
        the name of the graph, the second element is the corresponding
        embedding.
    """
    data = dataset.PathwayDataset(data_dir)
    emb_dir = os.path.abspath(os.path.join(data_dir, 'embeddings'))
    if not os.path.isdir(emb_dir):
        os.mkdir(emb_dir)


    if hyperparams is None:
        hyperparams = hyperparameters.get_hyperparameters()

    dim_latent = hyperparams['dim_latent']
    num_layers = hyperparams['num_layers']
    
    net = model.GCNModel(dim_latent=dim_latent, num_layers=num_layers)

    if load_model:
        model_path = os.path.abspath(os.path.join(data_dir, 'models/model.pth'))
        net.load_state_dict(torch.load(model_path))
    else:
        model_path = train.train(hyperparams=hyperparams, data_path=data_dir, plot=plot)
        net.load_state_dict(torch.load(model_path))

    embedding_dict = {}
    for idx in range(len(data)):
        graph, name = data[idx]
        embedding = net(graph)
        embedding_dict[name] = embedding
        if save:
            emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
            torch.save(embedding, emb_path)

    return embedding_dict


def get_embedding(name, data_dir='demo/data/example'):
    """Load and return the embedding of the graph with specified index."""
    emb_path = os.path.abspath(os.path.join(data_dir, 'embeddings'))
    embedding = torch.load(os.path.join(emb_path, f'{name}.pth'))
    return embedding


def embeddings_sanity_check(data_dir='demo/data/example'):
    """Check whether the embeddings of the toy examples are consistent
    between themselves. Works when embeddings have dimension = 1.

    Parameters
    ----------
    data_dir : str, optional
        Relative path to where the data is stored
    """
    studies = []
    for c in 'ABCD':
        studies.append(get_embedding(f'study_{c}'))
    stids = pickle.load(open(os.path.join(data_dir, 'info/sorted_stid_list.pkl', 'rb')))
    name_to_id = pickle.load(open(os.path.join(data_dir, 'info/name_to_id.pkl', 'rb')))
    test_1 = [studies[i][stids.index(name_to_id["Late endosomal microautophagy"])].item() for i in range(4)]
    assert test_1[0] == test_1[2] == test_1[3] != test_1[1]
    test_2 = [studies[i][stids.index(name_to_id["Signaling by NOTCH"])].item() for i in range(4)]
    assert test_2[0] == test_2[1] == test_2[3] != test_1[2]
    test_3 = [studies[i][stids.index(name_to_id["Macroautophagy"])].item() for i in range(4)]
    assert test_3[0] == test_3[2] == test_3[3] != test_3[1]
    assert (studies[0] == studies[3]).all()
    print("Passed all the tests!")
    return True


def nx_to_dgl(graph_nx):
    """Transform NetworkX graph into DGL graph."""
    for node in graph_nx.nodes:
        if graph_nx.nodes[node]['significance'] == 'significant':
            graph_nx.nodes[node]['significance'] = 1.0
        else:
            graph_nx.nodes[node]['significance'] = 0.0
    graph_dgl = dgl.from_networkx(graph_nx, node_attrs=['weight', 'significance'])
    return graph_dgl


def fit_cca_on_toy_data(data_dir='demo/data/example'):
    """Perform canonical correlation analysis on the toy datasets.

    Fit the CCA model onto the embeddings and labels of the studies
    A, B, and C. Study D is excluded since it is the same as A.

    Parameters
    ----------
    data_dir : str, optional
        Relative path to where the data is stored

    Returns
    -------
    abc.ABCMeta
        The trained CCA model which can be used on other embeddings
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

    emb_A = get_embedding('study_A', data_dir)
    emb_B = get_embedding('study_B', data_dir)
    emb_C = get_embedding('study_C', data_dir)
    emb_D = get_embedding('study_D', data_dir)

    stids = pickle.load(open(os.path.join(data_dir, 'info/sorted_stid_list.pkl'), 'rb'))
    name_to_id = pickle.load(open(os.path.join(data_dir, 'info/name_to_id.pkl'), 'rb'))

    indices_A = [stids.index(id) for name, id in name_to_id.items() if name in study_A]
    indices_B = [stids.index(id) for name, id in name_to_id.items() if name in study_B]
    indices_C = [stids.index(id) for name, id in name_to_id.items() if name in study_C]
    indices_D = [stids.index(id) for name, id in name_to_id.items() if name in study_D]

    y_A = torch.tensor([1.0 if i in indices_A else 0.0 for i in range(len(stids))]).unsqueeze(-1)
    y_B = torch.tensor([1.0 if i in indices_B else 0.0 for i in range(len(stids))]).unsqueeze(-1)
    y_C = torch.tensor([1.0 if i in indices_C else 0.0 for i in range(len(stids))]).unsqueeze(-1)
    y_D = torch.tensor([1.0 if i in indices_D else 0.0 for i in range(len(stids))]).unsqueeze(-1)

    cca = CCA(1)
    cca.fit(emb_A, y_A).fit(emb_B, y_B).fit(emb_C, y_C)
    return cca