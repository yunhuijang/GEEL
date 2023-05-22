import pickle
import networkx as nx
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from data.orderings import ORDER_FUNCS, order_graphs
from data.data_utils import train_val_test_split, adj_to_k2_tree, map_new_ordered_graph, adj_to_graph, tree_to_bfs_string
from data.mol_utils import canonicalize_smiles, smiles_to_mols, add_self_loop, tree_to_bfs_string_mol, mols_to_nx

from collections import Counter


with open('resource/GDSS_enz/GDSS_enz.pkl', 'rb') as f:
    graphs = pickle.load(f)


l = [len(list(nx.connected_components(graph))) for graph in graphs]
print(Counter(l))