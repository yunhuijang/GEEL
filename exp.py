import math
import torch
from treelib import Tree, Node
from collections import deque

from data.dataset import DATASETS
from data.orderings import order_graphs, ORDER_FUNCS

from data.data_utils import adj_to_k2_tree, tree_to_bfs_string, bfs_string_to_tree, get_k, get_level, get_parent, get_children_identifier

data_name = 'Caveman'
order = 'C-M'

graph_getter, num_rep = DATASETS[data_name]
graphs = graph_getter()
order_func = ORDER_FUNCS[order]
total_graphs = graphs
# print(len(graphs))
total_ordered_graphs = order_graphs(total_graphs, num_repetitions=num_rep, order_func=order_func, seed=0)
adjs = [graph.to_adjacency() for graph in total_ordered_graphs]
trees = [adj_to_k2_tree(adj, True) for adj in adjs]
strings = [tree_to_bfs_string(tree) for tree in trees]
print(max([len(string) for string in strings]))


# graph_sample = total_ordered_graphs[0]
# adj = graph_sample.to_adjacency()
# tree = adj_to_k2_tree(adj, True)


# max_len = 200
# d_model = 128


# position = torch.arange(max_len).unsqueeze(1)
# div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
# # pe: shape [max_len, 1, emb_size]
# pe = torch.zeros(max_len, 1, d_model)
# pe[:, 0, 0::2] = torch.sin(position * div_term)
# pe[:, 0, 1::2] = torch.cos(position * div_term)

# print(pe)