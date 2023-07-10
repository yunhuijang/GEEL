import math
import torch
import numpy as np
import networkx as nx


from data.orderings import bw_from_adj
from data.data_utils import load_graphs, get_max_len


from torch_geometric.datasets import MNISTSuperpixels

graphs = MNISTSuperpixels(root='resource', train=True)
print('hi')
# dataset_list = ['GDSS_ego', 'GDSS_com', 'GDSS_enz', 'GDSS_grid', 'planar', 'sbm']
# # dataset_list = ['proteins']
# order = 'C-M'
# for dataset_name in dataset_list:
#     print(get_max_len(dataset_name))
    # train_graphs, val_graphs, test_graphs = load_graphs(dataset_name, order)
    # bw = 0
    # for graphs in [train_graphs, val_graphs, test_graphs]:
    #     adjs = [nx.adjacency_matrix(graph) for graph in graphs]
    #     bw = max(max([bw_from_adj(adj.toarray()) for adj in adjs]), bw)
    # print(dataset_name)
    # print(bw)

# def bw_from_adj(A: np.ndarray) -> int:
#     """calculate bandwidth from adjacency matrix"""
#     band_sizes = np.arange(A.shape[0]) - A.argmax(axis=1)
#     return band_sizes.max()

# matrix = np.array([[1,1,1,0,0,0,0,0], [1,1,1,1,0,0,0,0], [1,1,1,1,1,0,0,0], [0,1,1,1,1,1,0,0],
#                   [0,0,1,1,1,1,1,0], [0,0,0,1,1,1,1,1], [0,0,0,0,1,1,1,1], [0,0,0,0,0,1,1,1]])

# print(bw_from_adj(matrix))