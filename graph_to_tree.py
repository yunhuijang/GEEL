from torch.nn import ZeroPad2d
from torch import LongTensor
from torch import count_nonzero
import math
from collections import deque
from time import time
import statistics

from graph_gen.data import DATASETS
from graph_gen.data.orderings import order_graphs, ORDER_FUNCS


def nearest_power_2(N):
    a = int(math.log2(N)) 
    if 2**a == N:
        return N

    return 2**(a + 1)

def adj_to_k2_tree(adj, k=4, is_wholetree=False):
    # TODO: generalize for other k
    n_org_nodes = adj.shape[0]
    # add padding (proper size for k)
    n_nodes = nearest_power_2(n_org_nodes)
    padder = ZeroPad2d((0, n_nodes-n_org_nodes, 0, n_nodes-n_org_nodes))
    padded_adj = padder(adj)
    k_sqrt = math.sqrt(k)
    tree_list = []
    leaf_list = []
    slice_size = int(n_nodes / k_sqrt)
    # slice matrices 
    sliced_adjs = deque([padded_adj[:slice_size, :slice_size], padded_adj[:slice_size, slice_size:], 
            padded_adj[slice_size:, :slice_size], padded_adj[slice_size:, slice_size:]])
    sliced_adjs_is_zero = LongTensor([int(count_nonzero(adj)>0) for adj in sliced_adjs])
    tree_list.append(sliced_adjs_is_zero)
    while (slice_size != 1):
        n_nodes = sliced_adjs[0].shape[0]
        if n_nodes == 2:
            if is_wholetree:
                leaf_list = [adj.reshape(4,) for adj in sliced_adjs]
            else:
                leaf_list = [adj.reshape(4,) for adj in sliced_adjs if count_nonzero(adj)>0]
            break
        slice_size = int(n_nodes / k_sqrt)
        target_adj = sliced_adjs.popleft()
        # remove adding leaves to 0
        if not is_wholetree:
            if count_nonzero(target_adj) == 0:
                continue
        new_sliced_adjs = [target_adj[:slice_size, :slice_size], target_adj[:slice_size, slice_size:], 
                target_adj[slice_size:, :slice_size], target_adj[slice_size:, slice_size:]]
        new_sliced_adjs_is_zero = LongTensor([int(count_nonzero(adj)>0) for adj in new_sliced_adjs])
        tree_list.append(new_sliced_adjs_is_zero)
        sliced_adjs.extend(new_sliced_adjs)
    return tree_list, leaf_list

def k2_tree_to_adj(tree_list, leaf_list):
    # TODO
    pass

def list_to_tree(tree_list, leaf_list):
    # TODO
    pass

# for data_name in ['GDSS_com', 'GDSS_enz', 'GDSS_grid', 'GDSS_ego']:
#     print(data_name)
#     for order in ['C-M', 'BFS', 'DFS']:
#         graph_getter, num_rep = DATASETS[data_name]
#         graphs = graph_getter()
#         order_func = ORDER_FUNCS[order]
#         total_graphs = graphs
#         total_ordered_graphs = order_graphs(total_graphs, num_repetitions=num_rep, order_func=order_func, seed=0)

#         graph_sample = total_ordered_graphs[0]
#         adj = graph_sample.to_adjacency()

#         adj_list = [graph.to_adjacency() for graph in total_ordered_graphs]
#         l = [adj_to_k2_tree(adj) for adj in adj_list]
#         tree_leaf_len_list = [4*(len(elem[0])+ len(elem[1])) for elem in l]
#         ratio_list = [length/((len(adj))**2) for length, adj in zip(tree_leaf_len_list, adj_list)]
#         print(f'{order}: {round(statistics.mean(ratio_list),3)}')
#     print(" ")

data_name = 'GDSS_ego'
order = 'C-M'
graph_getter, num_rep = DATASETS[data_name]
graphs = graph_getter()
order_func = ORDER_FUNCS[order]
total_graphs = graphs
total_ordered_graphs = order_graphs(total_graphs, num_repetitions=num_rep, order_func=order_func, seed=0)

graph_sample = total_ordered_graphs[0]
adj = graph_sample.to_adjacency()

adj_to_k2_tree(adj)