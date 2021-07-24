import os
import dgl
from reactome_gnn import dataset


ds = dataset.PathwayDataset(root='data/example')


def test_root_dir():
    assert os.path.isdir(ds.root)


def test_raw_dir():
    assert os.path.isdir(ds.raw_dir)


def test_save_dir():
    assert os.path.isdir(ds.save_dir)


def test_has_cache():
    assert ds.has_cache()


def test_getitem():
    graph, name = ds[0]
    assert isinstance(graph, dgl.DGLGraph) and isinstance(name, str)


def test_node_attributes():
    graph, name = ds[0]
    assert ('significance', 'weight') == tuple(graph.ndata.keys())
