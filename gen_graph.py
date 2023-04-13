import numpy as np
import networkx as nx
import os
import pickle
import json
from plot import plot_graphs_list
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import os.path as osp
import torch_geometric.transforms as T
import walker

def caveman_special(c=2,k=20,p_path=0.1,p_edge=0.3):
    p = p_path
    path_count = max(int(np.ceil(p * k)),1)
    G = nx.caveman_graph(c, k)
    # remove 50% edges
    p = 1-p_edge
    for (u, v) in list(G.edges()):
        if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
            G.remove_edge(u, v)
    # add path_count links
    for i in range(path_count):
        u = np.random.randint(0, k)
        v = np.random.randint(k, k * 2)
        G.add_edge(u, v)
    G = max([G.subgraph(c) for c in nx.connected_components(G)], key=len)
    return G



NAME_TO_NX_GENERATOR = {
    'grid': nx.generators.grid_2d_graph,  
    # -------- Additional datasets --------
    'gnp': nx.generators.fast_gnp_random_graph,  # fast_gnp_random_graph(n, p, seed=None, directed=False)
    'ba': nx.generators.barabasi_albert_graph,  # barabasi_albert_graph(n, m, seed=None)
    'pow_law': lambda **kwargs: nx.configuration_model(nx.generators.random_powerlaw_tree_sequence(**kwargs, gamma=3,
                                                                                                   tries=2000)),
    'except_deg': lambda **kwargs: nx.expected_degree_graph(**kwargs, selfloops=False),
    'cycle': nx.cycle_graph,
    'c_l': nx.circular_ladder_graph,
    'lobster': nx.random_lobster,
    'caveman': caveman_special
}

class GraphGenerator:
    def __init__(self, graph_type='grid', possible_params_dict=None, corrupt_func=None):
        if possible_params_dict is None:
            possible_params_dict = {}
        assert isinstance(possible_params_dict, dict)
        self.count = {k: 0 for k in possible_params_dict}
        self.possible_params = possible_params_dict
        self.corrupt_func = corrupt_func
        self.nx_generator = NAME_TO_NX_GENERATOR[graph_type]

    def __call__(self):
        params = {}
        for k, v_list in self.possible_params.items():
            params[k] = np.random.choice(v_list)
        graph = self.nx_generator(**params)
        graph = nx.relabel.convert_node_labels_to_integers(graph)
        if self.corrupt_func is not None:
            graph = self.corrupt_func(self.corrupt_func)
        return graph

def gen_graph_list(graph_type='grid', possible_params_dict=None, corrupt_func=None, length=1024, save_dir=None,
                   file_name=None, max_node=None, min_node=None):
    params = locals()
    if file_name is None:
        file_name = graph_type + '_' + str(length)
    file_path = os.path.join(save_dir, file_name)
    graph_generator = GraphGenerator(graph_type=graph_type,
                                     possible_params_dict=possible_params_dict,
                                     corrupt_func=corrupt_func)
    graph_list = []
    i = 0
    max_N = 0
    while i < length:
        graph = graph_generator()
        if max_node is not None and graph.number_of_nodes() > max_node:
            continue
        if min_node is not None and graph.number_of_nodes() < min_node:
            continue
        print(i, graph.number_of_nodes(), graph.number_of_edges())
        max_N = max(max_N, graph.number_of_nodes())
        if graph.number_of_nodes() <= 1:
            continue
        graph_list.append(graph)
        i += 1
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(file_path + '.pkl', 'wb') as f:
            pickle.dump(obj=graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(file_path + '.txt', 'w') as f:
            f.write(json.dumps(params))
            f.write(f'max node number: {max_N}')
    print(max_N)
    return graph_list

# gen_graph_list(graph_type='caveman', possible_params_dict={
#                                         'c': np.arange(2, 3).tolist(),
#                                         'k': np.arange(4, 6).tolist()}, 
#                                         corrupt_func=None, length=200, save_dir='resource', file_name='caveman')

dataset = 'Cora'
path = osp.join('../', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
G = to_networkx(data)
walks = walker.random_walks(G, n_walks=200, walk_len=100, start_nodes=[0])
graph_list = []
for walk in walks:
    new_graph = nx.Graph()
    new_graph.add_nodes_from(walk)
    new_edges = set([(walk[i], walk[i+1]) for i in range(len(walks[0])-1)])
    new_graph.add_edges_from(new_edges)
    graph_list.append(new_graph)

save_dir = 'resource'
file_name = 'cora'
file_path = os.path.join(save_dir, file_name)

with open(file_path + '.pkl', 'wb') as f:
    pickle.dump(obj=graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
with open(file_path + '.txt', 'w') as f:
    # f.write(json.dumps({'start_nodes: [0]'}))
    f.write(f'max node number: {200}, ')
    f.write(f'start_nodes: {[0]}')


print(len(graph_list))
print(max([len(graph.nodes) for graph in graph_list]))
print(min([len(graph.nodes) for graph in graph_list]))
plot_graphs_list(graph_list[:16], save_dir='cora', title='training')