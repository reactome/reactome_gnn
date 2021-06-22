import os
import math
import json
import urllib.request
from collections import defaultdict, deque, namedtuple

import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import matplotlib.pyplot as plt
from reactome2py import fiviz, analysis


class Marker:
    def __init__(self, marker_list, p_value, level):
        self.markers = ','.join(marker_list)
        self.p_value = p_value
        assert level in ('all', 'EHLD', 'SBGN'), \
            "Enrichment analysis pathway level must be 'all', 'EHLD', or 'SBGN'"
        self.level = level

        self.result = self.enrichment_analysis()
        if self.level in ('EHLD', 'SBGN'):
            self.currate_results()

    def enrichment_analysis(self):
        """
        Enrichment analysis performed on an entire graph (including Disease), return only those stIds whose p-value is less than 1.0.
        Here the user-defined threshold on p-value is taken into account, where all those pathways which have p_value <= threshold are considered significant,
        those that have p_value > threshold are insignificant (returned by enrichment analysis, but don't pass the user defined threshold).
        """
        result = analysis.identifiers(ids=self.markers)
        token = result['summary']['token']
        # Do I use the user-defined p-value here? 
        # If yes, then how do I differentiate significant from non-significant since only the significant pathways will be returned?
        token_result = analysis.token(token, species='Homo sapiens', page_size='-1', page='-1', sort_by='ENTITIES_FDR', 
                                      order='ASC', resource='TOTAL', p_value='1', include_disease=True, min_entities=None, 
                                      max_entities=None)
        info = [(p['stId'], p['entities']['pValue']) for p in token_result['pathways']]
        pathway_significance = {}
        for stid, p_val in info:
            significance = 'significant' if p_val < self.p_value else 'non-significant'
            pathway_significance[stid] = {'p_value': round(p_val, 4), 'significance': significance}
        return pathway_significance

    def currate_results(self):
        """
        In case a level for enrichment analysis is specified (EHLD or SBGN), the weights dictionary is modified so that only an intersection of stids that
        are both returned by enrichment analysis and are found at the specifeid level are considered significant/non-significant. Those that are not found
        at the specified level are considered 'not-found', regardles of whether they are returned by EA or not.
        """
        if self.level == 'EHLD':
            level_stids = set(fiviz.ehld_stids())
        if self.level == 'SBGN':
            level_stids == set(fiviz.sbgn_stids())

        for stid in self.result.keys():
            if stid not in level_stids:
                self.result[stid]['significance'] = 'not-found'


class Network:
    
    Info = namedtuple('Info', ['name', 'species', 'type', 'diagram'])

    def __init__(self, ea_results, study=None):
        self.study = study
        self.txt_url = 'https://reactome.org/download/current/ReactomePathwaysRelation.txt'
        self.json_url = 'https://reactome.org/ContentService/data/eventsHierarchy/9606'
        self.txt_adjacency = self.parse_txt(self.txt_url)
        self.json_adjacency, self.pathway_info, self.parent_dict = self.parse_json(self.json_url)
        self.weights = self.enrichment_analysis_weights(ea_results)
        self.name_to_id = self.set_name_to_id()
        self.node_attributes = self.set_node_attributes()
        self.graph_nx = self.to_networkx()
        
    def parse_txt(self, txt_url):
        """
        Given the URL to the TXT file, parse that file and create an adjacency list in the form of a dictionary.
        """
        txt_adjacency = defaultdict(list)
        found = False
        with urllib.request.urlopen(txt_url) as f:
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

    def parse_json(self, json_url):
        """
        Given the URL to the JSON file, parse that file and create an adjacency list in the form of a dictionary.
        """
        with urllib.request.urlopen(json_url) as f:
            tree_list = json.load(f)
        json_adjacency = defaultdict(list)
        parent_dict = defaultdict(list)
        pathway_info = {}
        for tree in tree_list:
            parent_dict[tree['stId']] = []
            self.recursive(tree, json_adjacency, pathway_info, parent_dict)
        parent_dict = dict(parent_dict)
        json_adjacency = dict(json_adjacency)
        return json_adjacency, pathway_info, parent_dict

    def recursive(self, tree, json_adjacency, pathway_info, parent_dict):
        """
        Considering that the JSON file has a nested structure, I wrote a recursive function that goes to the deepest level and build the tree from the bottom-up.
        Similar functionality could be achieved with DFS/BFS. Information for every node in the graph is stored in the pathway_info dictionary and each node's
        immediate parent is stored in the parent_dict dictionary.
        This is also true for the top-level pathways (roots of the trees), which have an empty list set as a parent
        (e.g. for Disease, parent_dict['R-HSA-1643685'] = []) and there is 28 of them.
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
            parent_dict[child['stId']].append(id)
            self.recursive(child, json_adjacency, pathway_info, parent_dict)

    def compare_graph(self):
        """
        Function that should show whether the txt-keys are a subset of json-keys but doesn't really work since some of the keys in the Disease pathways
        are not included in the JSON file, but are included in the TXT file. Or at least it was like that when I checked the last time, it might be different now.
        """
        if not set(self.txt_adjacency.keys()).issubset(self.json_adjacency.keys()):
            return False
        for key, value in self.txt_adjacency.items():
            if len(value) != len(set(value) & set(self.json_adjacency[key])):
                print(set(value) - set(self.json_adjacency[key]))
                print(set(self.json_adjacency[key]) - set(value))
                return False
        return True

    def enrichment_analysis_weights(self, ea_results):
        """
        Creates weights dictionary which includes information about all the human pathways, not only those returned by the enrichment analysis.
        Here the user-defined threshold on p-value is taken into account, where all those pathways which have p_value <= threshold are considered significant,
        those that have threshold < p_value <= 1.0 are insignificant (returned by enrichment analysis, but don't pass the user defined threshold),
        and those that are not returned by enrichment analysis are considered 'not-found'.

        Considereing that only some pathways are returned by analysis.identifiers function (those that have p_value < 1.0), how do I specify the p_value
        for all the other pathways in the graph? Here I put math.inf just as a placeholder.
        """
        weights = {}
        for stid in self.pathway_info.keys():
            if stid in ea_results.keys():
                weights[stid] = ea_results[stid].copy()
            else:
                weights[stid] = {'p_value': math.inf, 'significance': 'not-found'}
        return weights

    def set_node_attributes(self):
        node_attributes = {}
        for stid in self.pathway_info.keys():
            node_attributes[stid] = {
                'stid': stid,
                'name': self.pathway_info[stid].name,
                'weight': self.weights[stid]['p_value'],
                'significance': self.weights[stid]['significance'],   
            }
        return node_attributes

    def set_name_to_id(self):
        """
        Crete a dictionary which maps all the names to the stIds of the pathways, which is useful if we want to remove a pathway by giving its name
        instead of its stId. The problem is 
        """
        name_to_id = {}
        for id, info in self.pathway_info.items():
            name_to_id[info.name] = id
        return name_to_id

    def to_networkx(self, type='json'):
        graph_nx = nx.DiGraph()
        stids, names, weights, significance = {}, {}, {}, {}
        graph = self.txt_adjacency if type == 'txt' else self.json_adjacency
        for key, values in graph.items():
            for value in values:
                graph_nx.add_edge(key, value)

        for key, value in self.node_attributes.items():
            stids[key] = value['stid']
            names[key] = value['name']
            weights[key] = value['weight']
            significance[key] = value['significance']

        nx.set_node_attributes(graph_nx, stids, 'stId')
        nx.set_node_attributes(graph_nx, names, 'name')
        nx.set_node_attributes(graph_nx, significance, 'significance')
        nx.set_node_attributes(graph_nx, weights, 'weight')
        return graph_nx

    def remove_by_id(self, id):
        subtree = list(dfs_tree(self.graph_nx, id))
        self.graph_nx.remove_nodes_from(subtree)

    def remove_by_name(self, name):
        id = self.name_to_id[name]
        self.remove_by_id(id)
