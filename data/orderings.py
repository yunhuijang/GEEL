from typing import Callable
import random
from collections import deque
from operator import itemgetter
from dataclasses import dataclass

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix


def bw_from_adj(A: np.ndarray) -> int:
    """calculate bandwidth from adjacency matrix"""
    band_sizes = np.arange(A.shape[0]) - A.argmax(axis=1)
    return band_sizes.max()

def random_BFS_order(G: nx.Graph, seed=0) -> tuple:
    """
    :param G: Graph
    :return: random BFS order, maximum queue length (equal to bandwidth of ordering)
    """
    start = random.choice(list(G))
    visited = {start}
    queue = deque([start])
    max_q_len = 1
    order = []
    random.seed(seed)
    while queue:
        parent = queue.popleft()
        order.append(parent)
        children = sorted(set(G[parent]) - visited, key=lambda x: random.random())
        visited.update(children)
        queue.extend(children)
        max_q_len = max(len(queue), max_q_len)
    return order, max_q_len


def bw_from_order(G: nx.Graph, order: list) -> int:
    return bw_from_adj(nx.to_numpy_array(G, nodelist=order))


def random_DFS_order(G: nx.Graph, seed=0):
    """
    :param G: Graph
    :return: random DFS order, maximum queue length (equal to bandwidth of ordering)
    """
    start = random.choice(list(G))
    visited = {start}
    stack = [start]
    order = []
    random.seed(seed)
    while stack:
        parent = stack.pop()
        order.append(parent)
        children = sorted(set(G[parent]) - visited, key=lambda x: random.random())
        visited.update(children)
        stack.extend(children)
    bw = bw_from_order(G, order)
    return order, bw


def random_recursive_DFS_order(G: nx.Graph) -> tuple:
    visited = {}
    v = random.choice(list(G.nodes()))
    _recursive_DFS_order(G, v, visited)
    order = list(visited)
    bw = bw_from_order(G, order)
    return order, bw


def _recursive_DFS_order(G: nx.Graph, v, visited: dict):
    visited[v] = True  # use dict as ordered set
    if len(visited) == len(G):
        return
    children = sorted(set(G[v]) - set(visited), key=lambda x: random.random())
    for child in children:
        _recursive_DFS_order(G, child, visited)


def uniform_random_order(G: nx.Graph) -> tuple:
    order = list(G.nodes())
    random.shuffle(order)
    bw = bw_from_order(G, order)
    return order, bw


def random_connected_cuthill_mckee_ordering(G: nx.Graph, seed=0, heuristic=None) -> tuple:
    """
    adapted from NX source.
    :return: node order, bandwidth
    """
    # the cuthill mckee algorithm for connected graphs
    random.seed(seed)
    if heuristic is None:
        start = pseudo_peripheral_node(G, seed)
    else:
        start = heuristic(G)
    visited = {start}
    queue = deque([start])
    max_q_len = 1
    i = 0
    order = []
    while queue:
        parent = queue.popleft()
        order.append(parent)
        random.seed(seed+i)
        key = random.random()
        nd = sorted(list(G.degree(set(G[parent]) - visited)), key=lambda x: (x[1], key))
        children = [n for n, d in nd]
        visited.update(children)
        queue.extend(children)
        max_q_len = max(len(queue), max_q_len)
        i+=1
    return order, max_q_len


def pseudo_peripheral_node(G: nx.Graph, seed=0) -> int:
    """adapted from NX source"""
    # helper for cuthill-mckee to find a node in a "pseudo peripheral pair"
    # to use as good starting node
    random.seed(seed)
    u = random.choice(list(G))
    lp = 0
    v = u
    while True:
        spl = dict(nx.shortest_path_length(G, v))
        l = max(spl.values())
        if l <= lp:
            break
        lp = l
        farthest = (n for n, dist in spl.items() if dist == l)
        v, deg = min(G.degree(farthest), key=itemgetter(1))
    return v


@dataclass
class OrderedGraph:
    graph: nx.Graph
    seed: int
    ordering: list
    bw: int

    def to_data(self) -> Data:
        A = nx.to_scipy_sparse_matrix(self.graph, nodelist=self.ordering)
        edge_index = from_scipy_sparse_matrix(A)[0]
        return Data(edge_index=edge_index)

    def to_adjacency(self) -> torch.Tensor:
        return torch.tensor(
            nx.to_numpy_array(self.graph, nodelist=self.ordering),
            dtype=torch.float32,
        )


def order_graphs(
    graphs: list,
    order_func,
    num_repetitions: int = 1, seed: int = 0, is_mol=False
):
    ordered_graphs = []
    for i, graph in enumerate(graphs):
        for j in range(num_repetitions):
            # seed = i * (j + 1) + j
            random.seed(seed)
            np.random.seed(seed)
            if not is_mol:
                graph.remove_edges_from(nx.selfloop_edges(graph))
            graph = nx.convert_node_labels_to_integers(graph)
            order, bw = order_func(graph, seed)
            ordered_graphs.append(OrderedGraph(
                graph=graph, seed=seed,
                ordering=order, bw=bw,
            ))
    return ordered_graphs


ORDER_FUNCS = {
    "C-M": random_connected_cuthill_mckee_ordering,
    "BFS": random_BFS_order,
    "DFS": random_DFS_order,
}
