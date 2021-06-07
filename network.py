import os
import json
from collections import defaultdict, deque, namedtuple

import networkx as nx
import matplotlib.pyplot as plt


class Network:
    
    Info = namedtuple('Info', ['name', 'species', 'type', 'diagram'])

    def __init__(self, txt_path='ReactomePathwaysRelation.txt', json_path='9606'):
        self.graph_txt = self.parse_txt(txt_path)
        self.graph_dict, self.pathway_info, self.parent_dict = self.parse_json(json_path)
        self.name_to_id = self.get_name_to_id()
        self.graph_nx = None
        # TODO: After removal of a subtree, graph_nx is not updated
        
    def parse_txt(self, txt_path):
        self.graph_txt = defaultdict(list)
        found = False
        with open(txt_path) as f:
            lines = f.readlines()
            for line in lines:
                stid1, stid2 = line.strip().split()
                if not 'R-HSA' in stid1:
                    if found:
                        break
                    else:
                        continue
                self.graph_txt[stid1].append(stid2)
        return self.graph_txt

    def parse_json(self, json_path):
        with open(json_path) as f:
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

    def to_networkx(self, type='txt'):
        self.graph_nx = nx.DiGraph()
        graph = self.graph_txt if type == 'txt' else self.graph_dict
        for key, values in graph.items():
            for value in values:
                self.graph_nx.add_edge(key, value)
        return self.graph_nx

    def visualize(self, fig_path='graph.png'):
        if self.graph_nx is None:
            self.to_networkx()
        nx.draw(self.graph_nx, node_size=2, width=0.2, arrowsize=2)
        plt.savefig(fig_path)