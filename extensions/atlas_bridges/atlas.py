"""A small bridge-detection task built from the NetworkX Graph Atlas."""

import random

import networkx as nx
import torch
from torch_geometric.data import Data


class AtlasGraphs(list):
    # The group trainer asks its dataset for these two dimensions.
    num_node_features = 3
    num_classes = 2


def load_graphs():
    graphs = AtlasGraphs()
    catalog = []
    for atlas_id, graph in enumerate(nx.graph_atlas_g()):
        if len(graph) != 7 or not nx.is_connected(graph):
            continue
        has_bridge = int(nx.has_bridges(graph))
        degrees = [graph.degree(node) for node in range(7)]
        features = [[1.0, degree / 6, 0.0] for degree in degrees]
        edges = torch.tensor(list(graph.to_directed().edges), dtype=torch.long).T.contiguous()
        graphs.append(Data(x=torch.tensor(features), edge_index=edges,
                           y=torch.tensor([has_bridge]), atlas_id=atlas_id))
        catalog.append({
            "atlas_id": atlas_id,
            "graph6": nx.to_graph6_bytes(graph, header=False).decode().strip(),
            "has_bridge": has_bridge,
            "has_leaf": int(1 in degrees),
        })
    return graphs, catalog


def add_cue(graphs, seed):
    """Make a marked copy. The random draws are separate from model and split RNGs."""
    rng = random.Random(10_000 + seed)
    marked = AtlasGraphs()
    for graph in graphs:
        copy = graph.clone()
        sign = 2 * graph.y.item() - 1
        copy.x[:, 2] = sign if rng.random() < 0.95 else -sign
        marked.append(copy)
    return marked
