import os
import json
from collections import defaultdict, namedtuple

import networkx as nx
import matplotlib.pyplot as plt


Info = namedtuple('Info', ['name', 'species', 'type', 'diagram'])


def to_networkx(graph_dict):
    graph_nx = nx.DiGraph()
    for key, values in graph_dict.items():
        for value in values:
            graph_nx.add_edge(key, value)
    return graph_nx


def visualize(graph_nx, fig_path='graph.png'):
    nx.draw(graph_nx, node_size=2, width=0.2, arrowsize=2)
    plt.savefig(fig_path)


def parse_text(path='ReactomePathwaysRelation.txt'):
    graph = defaultdict(list)
    found = False
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            stid1, stid2 = line.strip().split()
            if not 'R-HSA' in stid1:
                if found:
                    break
                else:
                    continue
            graph[stid1].append(stid2)
    return graph


def recursive(tree, graph_dict, pathway_info, parent_dict):
    id = tree['stId']
    try:
        pathway_info[id] = Info(tree['name'], tree['species'], tree['type'], tree['diagram'])
    except KeyError:
        pathway_info[id] = Info(tree['name'], tree['species'], tree['type'], None)
    try:
        children = tree['children']
    except KeyError:
        return
    for child in children:
        graph_dict[id].append(child['stId'])
        parent_dict[child['stId']].append(id)
        recursive(child, graph_dict, pathway_info, parent_dict)


def parse_json(path='9606'):
    with open(path) as f:
        tree_list = json.load(f)
    graph_dict = defaultdict(list)
    parent_dict = defaultdict(list)
    pathway_info = {}
    for tree in tree_list:
        parent_dict[tree['stId']] = [None]
        recursive(tree, graph_dict, pathway_info, parent_dict)
    return graph_dict, pathway_info, parent_dict


def compare_graph(graph_txt, graph_json):
    # txt-keys should be a subset of json-keys
    if not set(graph_txt.keys()).issubset(graph_json.keys()):
        return False
    for key, value in graph_txt.items():
        if len(value) != len(set(value) & set(graph_json[key])):
            print(set(value) - set(graph_json[key]))
            print(set(graph_json[key]) - set(value))
            return False
    return True


def remove_by_id(graph_dict, parent_dict, id):
    # Optimize with deque instead of list
    queue = []
    visited = []
    visited.append(id)
    queue.append(id)
    while queue:
        s = queue.pop(0)
        for child in graph_dict[s]:
            if child not in visited:
                visited.append(child)
                queue.append(child)
    for node in reversed(visited):
        for parent in parent_dict[node]:
            try:
                graph_dict[parent].remove(node)
            except:
                pass  # The parent node has already been removed
        graph_dict.pop(node)


def remove_by_name(graph_dict, parent_dict, name, name_to_id):
    id = name_to_id[name]
    remove_by_id(graph_dict, parent_dict, id)
    

def get_name_to_id(pathway_info):
    name_to_id = {}
    for id, info in pathway_info.items():
        name_to_id[info.name] = id
    return name_to_id


if __name__ == '__main__':
    graph_txt = parse_text()
    # print(len(graph.keys()))
    graph_json, pathway_info, parent_dict = parse_json()
    for key, value in parent_dict.items():
        if len(value) > 1:
            print(key)
            print(value)
    print(pathway_info['R-HSA-432712'])
    # print(graph)
    print(compare_graph(graph_txt, graph_json))
