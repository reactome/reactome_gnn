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
        Given the URL to the TXT file, parse that file and create an adjacency list in the form of a dictionary.
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
        Given the URL to the JSON file, parse that file and create an adjacency list in the form of a dictionary.
        It also creates a pathway_info dictionary in which the information from the JSON file is stored for each node.
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
        """
        A recursive function that parser the nested JSON file and builds that graph form the bottom up. 
        Similar functionality could be achieved with DFS/BFS. However in the previous version of code the parents
        (predecessors) for each node were also stored in a dictionary, which helped with the node removal.
        Considering that now the node removal is done through NetworkX, there is no need for parents dictionary.
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
        """
        Function that shows whether the txt-keys are a subset of json-keys.
        However, some of the nodes form the Disease tree are not included in the JSON file,
        but are included in the TXT file.
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
        """
        Creates weights dictionary which includes information for all the human pathways.
        Those pathways which are returned by the enrichment analysis simply get their p-value and significance copied.
        Those pathways that were not returned, have their significance set to 'not-found'.

        ----------------------------------
        How do I specifiy the p-value for the not-found pathways? Maybe 1.0?
        Here I put math.inf just as a placeholder.
        ----------------------------------
        """
        weights = {}
        for stid in self.pathway_info.keys():
            if stid in ea_result.keys():
                weights[stid] = ea_result[stid].copy()
            else:
                weights[stid] = {'p_value': math.inf, 'significance': 'not-found'}
        return weights

    def set_node_attributes(self):
        """
        Saves the required node attributes requires for the NetworkX graph into a dictionaries.
        Helps with creating the NetworkX object.
        """
        stids, names, weights, significances = {}, {}, {}, {}
        for stid in self.pathway_info.keys():
            stids[stid] = stid
            names[stid] = self.pathway_info[stid].name
            weights[stid] = 1.0 if self.weights is None else self.weights[stid]['p_value']
            significances[stid] = 'not-found' if self.weights is None else self.weights[stid]['significance']
        return stids, names, weights, significances

    def set_name_to_id(self):
        """
        Cretes a dictionary which maps all the names to the stIds of the pathways.
        This is useful for when we want to remove a node by specifying a pathway's name instead of ID.

        ----------------------------------
        Issue: some nodes have the same name but different ID.
        ----------------------------------
        """
        name_to_id = {}
        for id, info in self.pathway_info.items():
            name_to_id[info.name] = id
        return name_to_id

    def to_networkx(self, type='json'):
        """
        Creates the NetworkX DiGraph by iterating over an adjecancy list created from parsing the JSON file.
        """
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
        """
        Removes the subtree which has the specified stId as its root.
        """
        subtree = list(dfs_tree(self.graph_nx, stid))
        self.graph_nx.remove_nodes_from(subtree)

    def remove_by_name(self, name):
        """
        Removes the subtree which has the specified name as its root.
        """
        id = self.name_to_id[name]
        self.remove_by_id(id)

    def add_significance_by_stid(self, stid_list):
        for stid in stid_list:
            try:
                self.graph_nx.nodes[stid]['significance'] = 'significant'
                self.graph_nx.nodes[stid]['weight'] = 0.0
            except KeyError:
                continue

    def add_significance_by_name(self, name_list):
        stid_list = [self.name_to_id[name] for name in name_list]
        self.add_significance_by_stid(stid_list)
