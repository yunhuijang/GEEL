import torch
from torch.nn import ZeroPad2d
from torch import LongTensor
from torch.utils.data import random_split
from torch import count_nonzero
import math
from collections import deque
from treelib import Tree, Node
from sklearn.model_selection import train_test_split
from itertools import zip_longest
import networkx as nx
from treelib import Tree, Node
import numpy as np
from itertools import compress, islice
import os
from pathlib import Path
import json


from data.tokens import grouper_mol


DATA_DIR = "resource"
NODE_TYPE_DICT = {'F': 9, 'O': 10, 'N': 11, 'C': 12, 'P': 13, 'I': 14, 'Cl': 15, 'Br': 16, 'S': 17}
TYPE_NODE_DICT = {str(key): value for value, key in NODE_TYPE_DICT.items()}
BOND_TYPE_DICT = {1: 5, 2: 6, 3: 7, 1.5: 8}
TYPE_BOND_DICT = {key: value for value, key in NODE_TYPE_DICT.items()}


def adj_to_adj_list(adj):
    # TODO
    '''
    adjacency matrix to adjacency list
    '''
    adj_list = 0
    return adj_list


def adj_list_to_adj(adj_list):
    # TODO
    '''
    adjacency list to adjacency matrix
    '''
    adj = 0
    return adj

    
def train_val_test_split(
    data: list,
    data_name='GDSS_com',
    train_size: float = 0.7, val_size: float = 0.1, test_size: float = 0.2,
    seed: int = 42,
):
    if data_name in ['qm9', 'zinc']:
        # code adpated from https://github.com/harryjo97/GDSS
        with open(os.path.join(DATA_DIR, f'{data_name}/valid_idx_{data_name}.json')) as f:
            test_idx = json.load(f)
        if data_name == 'qm9':
            test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]
        train_idx = [i for i in range(len(data)) if i not in test_idx]
        test = [data[i] for i in test_idx]
        train_val = [data[i] for i in train_idx]
        train, val = train_test_split(train_val, train_size=train_size / (train_size + val_size), random_state=seed, shuffle=True)
    elif data_name in ['planar', 'sbm', 'proteins']:
        # code adapted from https://github.com/KarolisMart/SPECTRE
        test_len = int(round(len(data)*0.2))
        train_len = int(round((len(data) - test_len)*0.8))
        val_len = len(data) - train_len - test_len
        train, val, test = random_split(data, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))
    else:
        train_val, test = train_test_split(data, train_size=train_size + val_size, shuffle=False)
        train, val = train_test_split(train_val, train_size=train_size / (train_size + val_size), random_state=seed, shuffle=True)
    return train, val, test

def adj_to_graph(adj, is_cuda=False):
    '''
    adjacency matrix to graph
    '''
    if is_cuda:
        adj = adj.detach().cpu().numpy()
    G = nx.from_numpy_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() < 1:
        G.add_node(1)
    return G
                
def map_new_ordered_graph(ordered_graph):
    '''
    Map ordered_graph object to ordered networkx graph
    '''
    org_graph = ordered_graph.graph
    ordering = ordered_graph.ordering
    mapping = {i: ordering.index(i) for i in range(len(ordering))}
    new_graph = nx.relabel_nodes(org_graph, mapping)
    return new_graph

def fix_symmetry(adj):
    sym_adj = torch.tril(adj) + torch.tril(adj).T
    return torch.where(sym_adj>0, 1, 0)

# def get_max_len(data_name, order='C-M', k=2):
#     total_strings = []
#     k_square = k**2
#     for split in ['train', 'test', 'val']:
#         string_path = os.path.join(DATA_DIR, f"{data_name}/{order}/{data_name}_str_{split}_{k}.txt")
#         strings = Path(string_path).read_text(encoding="utf=8").splitlines()
        
#         total_strings.extend(strings)
    
    
#     max_len = max([len(string) for string in total_strings])
#     group_max_len = max_len / k_square
    
#     return max_len, group_max_len
    