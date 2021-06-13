import os
import json
import urllib.request
from collections import defaultdict, deque, namedtuple

import networkx as nx
import matplotlib.pyplot as plt
from reactome2py import fiviz, analysis


class Network:
    
    Info = namedtuple('Info', ['name', 'species', 'type', 'diagram'])

    def __init__(self, txt_url='https://reactome.org/download/current/ReactomePathwaysRelation.txt',
                 json_url='https://reactome.org/ContentService/data/eventsHierarchy/9606'):
        self.graph_txt = self.parse_txt(txt_url)
        self.graph_dict, self.pathway_info, self.parent_dict = self.parse_json(json_url)
        self.name_to_id = self.get_name_to_id()
        self.markers = 'RAS,MAP,IL10,EGF,EGFR,STAT'
        self.weights = {key: 0 for key in self.pathway_info.keys()}
        self.node_attributes = {}
        self.graph_nx = None
        # TODO: After removal of a subtree, graph_nx is not updated

    def set_node_attributes(self):
        for key in self.pathway_info.keys():
            self.node_attributes[key] = {
                'stid': key,
                'name': self.pathway_info[key].name,
                'parent': self.parent_dict[key],
                'weight': self.weights[key]
            }
        
    def parse_txt(self, txt_url):
        self.graph_txt = defaultdict(list)
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
                self.graph_txt[stid1].append(stid2)
        return self.graph_txt

    def parse_json(self, json_url):
        with urllib.request.urlopen(json_url) as f:
            tree_list = json.load(f)
        graph_dict = defaultdict(list)
        parent_dict = defaultdict(list)
        pathway_info = {}
        for tree in tree_list:
            parent_dict[tree['stId']] = [None]
            self.recursive(tree, graph_dict, pathway_info, parent_dict)
        return graph_dict, pathway_info, parent_dict

    def recursive(self, tree, graph_dict, pathway_info, parent_dict):
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
            graph_dict[id].append(child['stId'])
            parent_dict[child['stId']].append(id)
            self.recursive(child, graph_dict, pathway_info, parent_dict)

    def compare_graph(self):
        # txt-keys should be a subset of json-keys
        if not set(self.graph_txt.keys()).issubset(self.graph_dict.keys()):
            return False
        for key, value in self.graph_txt.items():
            if len(value) != len(set(value) & set(self.graph_dict[key])):
                print(set(value) - set(self.graph_dict[key]))
                print(set(self.graph_dict[key]) - set(value))
                return False
        return True

    def remove_by_id(self, id):
        queue = deque([id])
        visited = [id]
        while queue:
            s = queue.popleft()
            for child in self.graph_dict[s]:
                if child not in visited:
                    visited.append(child)
                    queue.append(child)
        for node in reversed(visited):
            for parent in self.parent_dict[node]:
                try:
                    self.graph_dict[parent].remove(node)
                except:
                    pass  # The parent node has already been removed or doesn't exist (top pathway)
            self.graph_dict.pop(node)

    def remove_by_name(self, name):
        id = self.name_to_id[name]
        self.remove_by_id(id)

    def get_name_to_id(self):
        self.name_to_id = {}
        for id, info in self.pathway_info.items():
            self.name_to_id[info.name] = id
        return self.name_to_id

    def to_networkx(self, type='json'):
        self.graph_nx = nx.DiGraph()
        stids, names, parents, weights = {}, {}, {}, {}
        graph = self.graph_txt if type == 'txt' else self.graph_dict
        for key, values in graph.items():
            for value in values:
                self.graph_nx.add_edge(key, value)

        # TODO: Optimize this, too many of those values are calculated multiple times
        for key, value in self.node_attributes.items():
            stids[key] = value['stid']
            names[key] = value['name']
            parents[key] = value['parent']
            weights[key] = value['weight']

        nx.set_node_attributes(self.graph_nx, stids, 'stid')
        nx.set_node_attributes(self.graph_nx, names, 'name')
        nx.set_node_attributes(self.graph_nx, parents, 'parent')
        nx.set_node_attributes(self.graph_nx, weights, 'weight')
        return self.graph_nx

    def visualize(self, fig_path='data/graph.png'):
        if self.graph_nx is None:
            self.to_networkx()
        nx.draw(self.graph_nx, node_size=2, width=0.2, arrowsize=2)
        plt.savefig(fig_path)

    def enrichment_analysis(self):        
        result = analysis.identifiers(ids=self.markers, interactors=False, page_size='1', page='1', species='Homo Sapiens', sort_by='ENTITIES_FDR', order='ASC', resource='TOTAL', p_value='1', include_disease=False, min_entities=None, max_entities=None, projection=True)
        token = result['summary']['token']
        token_result = analysis.token(token, species='Homo sapiens', page_size='-1', page='-1', sort_by='ENTITIES_FDR', order='ASC', resource='TOTAL', p_value='1', include_disease=False, min_entities=None, max_entities=None)
        return [p['stId'] for p in token_result['pathways']]

    def set_weights(self, mode):
        assert mode in (1, 2, 3), "Mode has to be 1, 2, or 3."
        if mode == 1:
            stids = self.enrichment_analysis()
        if mode == 2:
            stids = fiviz.ehld_stids()
        if mode == 3:
            stids = fiviz.sbgn_stids()
        self.weights = {key: 1 if key in stids else 0 for key in self.pathway_info.keys()}
