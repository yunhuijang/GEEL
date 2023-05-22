import pickle
import networkx as nx
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

from data.orderings import ORDER_FUNCS, order_graphs
from data.data_utils import train_val_test_split, adj_to_k2_tree, map_new_ordered_graph, adj_to_graph, tree_to_bfs_string
from data.mol_utils import canonicalize_smiles, smiles_to_mols, add_self_loop, tree_to_bfs_string_mol, mols_to_nx

from collections import Counter

from data.data_utils import get_max_len, remove_redundant, generate_final_tree_red, generate_initial_tree_red, fix_symmetry, tree_to_adj, grouper
from plot import plot_one_graph

from networkx import adjacency_matrix

# print(get_max_len('GDSS_com', order='C-M', k=3))
# print(get_max_len('planar', order='C-M', k=3))
with open('gcg/resource/GDSS_com/C-M/GDSS_com_test_graphs.pkl', 'rb') as f:
    graphs = pickle.load(f)
org_graph = graphs[15]
adj_org = nx.adjacency_matrix(org_graph)
tree_org = adj_to_k2_tree(torch.Tensor(adj_org.todense()), return_tree=True, k=3, is_mol=False)
string_org = tree_to_bfs_string(tree_org, string_type='group')
# strings = Path('gcg/resource/GDSS_com/C-M/GDSS_com_str_test_3.txt').read_text(encoding="utf=8").splitlines()

string = remove_redundant(string_org, False, k=3)
print(string)
tree = generate_initial_tree_red(string, k=3)
valid_tree = generate_final_tree_red(tree, k=3)

adj = fix_symmetry(tree_to_adj(valid_tree, k=3))
graph = adj_to_graph(adj.numpy())
print(graph)