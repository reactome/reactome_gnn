import math
import json
import urllib.request
from collections import defaultdict, namedtuple
from datetime import datetime

import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree


class Network:
    
    Info = namedtuple('Info', ['name', 'species', 'type', 'diagram'])

    def __init__(self, ea_result=None, study=None):
        self.txt_url = 'https://reactome.org/download/current/ReactomePathwaysRelation.txt'
        self.json_url = 'https://reactome.org/ContentService/data/eventsHierarchy/9606'
        if study is not None:
            self.study = study
        else:
            time_now = datetime.now().strftime('%Y-%b-%d-%H-%M')
            study = time_now
        self.txt_adjacency = self.parse_txt()
        self.json_adjacency, self.pathway_info = self.parse_json()
        if ea_result is not None:
            self.weights = self.set_weights(ea_result)
        else:
            self.weights = None
        self.name_to_id = self.set_name_to_id()
        self.graph_nx = self.to_networkx()
        
    def parse_txt(self):
        """
        Given the URL to the TXT file, parse that file and create an
        adjacency list in the form of a dictionary.

        Returns
        -------
        dict
            Adjacency list for the graph parsed from the TXT file
        """
        txt_adjacency = defaultdict(list)
        found = False
        with urllib.request.urlopen(self.txt_url) as f:
            lines = f.readlines()
            for line in lines:
                line = line.decode('utf-8')
                stid1, stid2 = line.strip().split()
                if not 'R-HSA' in stid1:
                    if found:
                        break
                    else:
                        continue
                txt_adjacency[stid1].append(stid2)
        txt_adjacency = dict(txt_adjacency)
        return txt_adjacency

    def parse_json(self):
        """
        Given the URL to the JSON file, parse that file and create an
        adjacency list in the form of a dictionary. It also creates a
        pathway_info dictionary in which the information from the JSON
        file is stored for each node.

        Returns
        -------
        dict
            Adjacency list for the graph parsed from the JSON file
        dict
            Dictionary where information about each node is stored
        """
        with urllib.request.urlopen(self.json_url) as f:
            tree_list = json.load(f)
        json_adjacency = defaultdict(list)
        pathway_info = {}
        for tree in tree_list:
            self.recursive(tree, json_adjacency, pathway_info)
        json_adjacency = dict(json_adjacency)
        return json_adjacency, pathway_info

    def recursive(self, tree, json_adjacency, pathway_info):
        """A recursive function that parser the nested JSON file.

        Parameters
        ----------
        tree : dict
            Nested dictionary obtained by parsing the JSON file
        json_adjacency : dict
            Adjacency list for the graph parsed from the JSON file
        pathway_info : dict
            Dictionary where information about each node is stored
        
        Returns
        -------
        None
        """
        id = tree['stId']
        try:
            pathway_info[id] = Network.Info(tree['name'], tree['species'], tree['type'], tree['diagram'])
        except KeyError:
            pathway_info[id] = Network.Info(tree['name'], tree['species'], tree['type'], None)
        try:
            children = tree['children']
        except KeyError:
            return
        for child in children:
            json_adjacency[id].append(child['stId'])
            self.recursive(child, json_adjacency, pathway_info)

    def compare_graphs(self):
        """Return whether the txt-keys are a subset of json-keys.

        **Note**:
        Some of the nodes from the Disease tree are not
        included in the JSON file, but are included in the TXT file.
        Which is why this function doesn't work perferctly.

        Returns
        -------
        bool
            Whether the TXT-graph is a subset of JSON-graph
        """
        if not set(self.txt_adjacency.keys()).issubset(self.json_adjacency.keys()):
            return False
        for key, value in self.txt_adjacency.items():
            if len(value) != len(set(value) & set(self.json_adjacency[key])):
                print(set(value) - set(self.json_adjacency[key]))
                print(set(self.json_adjacency[key]) - set(value))
                return False
        return True

    def set_weights(self, ea_result):
        """Set weights for each node in the human pathway graph.
        
        The pathways which are returned by the enrichment analysis
        simply get their p-value and significance copied. The pathways
        that were not returned, have their significance set to
        'not-found'.

        Parameters
        ----------
        ea_results : dict
            Dictionary where the p-value and significance are stored
            for each pathways returned by the enrichment analysis.

        Returns
        -------
        dict
            Expanded dictionary with weights for all the pathways in
            the human pathway network, instead of just those returned
            by the enrichment nanalysis
        """
        weights = {}
        for stid in self.pathway_info.keys():
            if stid in ea_result.keys():
                weights[stid] = ea_result[stid].copy()
            else:
                weights[stid] = {'p_value': 1.0, 'significance': 'not-found'}
        return weights

    def set_node_attributes(self):
        """Return dictionaries with node attributes for NetworkX graph.

        Four values are used as node attributes in the NetworkX graph
        ---stid, name, weight, significance. These values are stored
        into four different dictionaries for all the nodes in the
        graph, which helps with creation of the NetworkX object.

        Returns
        -------
        dict
            Dictionary where stids of all the nodes are stored
        dict
            Dictionary where names of all the nodes are stored
        dict
            Dictionary where weights of all the nodes are stored
        dict
            Dictionary where significances of all the nodes are stored
        """
        stids, names, weights, significances = {}, {}, {}, {}
        for stid in self.pathway_info.keys():
            stids[stid] = stid
            names[stid] = self.pathway_info[stid].name
            weights[stid] = 1.0 if self.weights is None else self.weights[stid]['p_value']
            significances[stid] = 'not-found' if self.weights is None else self.weights[stid]['significance']
        return stids, names, weights, significances

    def set_name_to_id(self):
        """Return dictionary mapping the names to stids

        **Note**:
        Issue: some nodes have the same name but different ID.

        Returns
        -------
        dict
            Dictionary where the names of the nodes are keys, and
            the stids are the values
        """
        name_to_id = {}
        for id, info in self.pathway_info.items():
            name_to_id[info.name] = id
        return name_to_id

    def to_networkx(self, type='json'):
        """Creates the NetworkX DiGraph by from the json-adjacency."""
        graph_nx = nx.DiGraph()
        graph = self.json_adjacency if type == 'json' else self.txt_adjacency
        for key, values in graph.items():
            for value in values:
                graph_nx.add_edge(key, value)

        stids, names, weights, significances = self.set_node_attributes()

        nx.set_node_attributes(graph_nx, stids, 'stId')
        nx.set_node_attributes(graph_nx, names, 'name')
        nx.set_node_attributes(graph_nx, weights, 'weight')
        nx.set_node_attributes(graph_nx, significances, 'significance')

        return graph_nx

    def remove_by_id(self, stid):
        """Removes the subtree which has the specified stid as its root."""
        subtree = list(dfs_tree(self.graph_nx, stid))
        self.graph_nx.remove_nodes_from(subtree)

    def remove_by_name(self, name):
        """Removes the subtree which has the specified name as its root."""
        id = self.name_to_id[name]
        self.remove_by_id(id)

    def add_significance_by_stid(self, stid_list):
        """Highlight only the pathways with the specified stids.
        
        **Note**:
        How to add weights (p-values) for the specified nodes?  
        """
        for stid in stid_list:
            try:
                self.graph_nx.nodes[stid]['significance'] = 'significant'
                self.graph_nx.nodes[stid]['weight'] = 0.0
            except KeyError:
                continue

    def add_significance_by_name(self, name_list):
        """Highlight only the pathways with the specified names."""
        stid_list = [self.name_to_id[name] for name in name_list]
        self.add_significance_by_stid(stid_list)
