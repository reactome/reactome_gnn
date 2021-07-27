import os
import pickle
import subprocess

import dgl
from dgl.data import DGLDataset


class PathwayDataset(DGLDataset):
    """
    A class that inherits from DGLDataset and extends its functionality
    by adding additional attributed and processing of the graph
    accordingly.

    Attributes
    ----------
    root : str
        Root directory consisting of other directories where the raw
        data can be found, and where all the processing results are
        stored.
    """

    def __init__(self, root='data'):
        """
        Parameters
        ----------
        root : str
            Root directory consisting of other directories where the raw
            data can be found, and where all the processing results are
            stored.
        """
        self.root = os.path.abspath(root)
        if 'processed' not in os.listdir(self.root):
            subprocess.run(f"mkdir 'tmp'", shell=True, cwd=self.root)
        raw_dir = os.path.join(self.root, 'raw')
        save_dir = os.path.join(self.root, 'processed')
        super().__init__(name='pathway_graph', raw_dir=raw_dir, save_dir=save_dir)
        

    def has_cache(self):
        """Check whether the dataset already exists."""
        return len(os.listdir(self.save_dir)) == len(os.listdir(self.raw_dir))

    def __len__(self):
        return len(os.listdir(self.save_dir))

    def __getitem__(self, idx):
        names = sorted(os.listdir(self.save_dir))
        name = names[idx]
        (graph,), _ = dgl.load_graphs(os.path.join(self.save_dir, name))
        return graph, name

    def process(self):
        """Process the graphs and store them in the 'processed' directory."""
        for cnt, graph_file in enumerate(os.listdir(self.raw_dir)):
            graph_path = os.path.join(self.raw_dir, graph_file)
            nx_graph = pickle.load(open(graph_path, 'rb'))
            for node in nx_graph.nodes:
                if nx_graph.nodes[node]['significance'] == 'significant':
                    nx_graph.nodes[node]['significance'] = 1.0
                else:
                    nx_graph.nodes[node]['significance'] = 0.0
            dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['weight', 'significance'])
            save_path = os.path.join(self.save_dir, f'{graph_file[:-4]}.dgl')
            dgl.save_graphs(save_path, dgl_graph)