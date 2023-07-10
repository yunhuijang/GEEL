import torch
from torch.utils.data import random_split
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import networkx as nx
import numpy as np
import os
import json
import math

from data.orderings import ORDER_FUNCS, order_graphs


DATA_DIR = "resource"
NODE_TYPE_DICT = {'F': 9, 'O': 10, 'N': 11, 'C': 12, 'P': 13, 'I': 14, 'Cl': 15, 'Br': 16, 'S': 17}
TYPE_NODE_DICT = {str(key): value for value, key in NODE_TYPE_DICT.items()}
BOND_TYPE_DICT = {1: 5, 2: 6, 3: 7, 1.5: 8}
TYPE_BOND_DICT = {key: value for value, key in NODE_TYPE_DICT.items()}


def adj_to_adj_list(adj):
    '''
    adjacency matrix to adjacency list
    '''
    adj_matrix = adj.todense()
    num_nodes = len(adj_matrix)
    adj_list = []
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i,j] == 1:
                adj_list.append((i, j))
    
    return sorted([(j,i) for i, j in adj_list])


def adj_list_to_adj(adj_list):
    '''
    adjacency list to adjacency matrix
    '''
    if len(adj_list) < 2:
        num_nodes = len(adj_list)
        adj = [[0] * num_nodes for _ in range(num_nodes)]
        return adj
    
    num_nodes =  max(map(max, adj_list))+1
    adj = [[0] * num_nodes for _ in range(num_nodes)]
    
    for n, e in adj_list:
        adj[n][e] = 1
        adj[e][n] = 1

    return adj

def adj_list_diff_to_adj_list(adj_list_diff):
    return [(token[0], token[0]+token[1]) for token in adj_list_diff]

    
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
    if isinstance(adj, (np.ndarray, np.generic)):
        G = nx.from_numpy_matrix(adj)
    else:
        G = nx.from_numpy_matrix(adj.numpy())
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
    new_graph = nx.from_numpy_array(nx.adjacency_matrix(org_graph, nodelist=ordering))
    return new_graph

def fix_symmetry(adj):
    sym_adj = torch.tril(adj) + torch.tril(adj).T
    return torch.where(sym_adj>0, 1, 0)

# codes adapted from https://github.com/KarolisMart/SPECTRE
def load_proteins_data(data_dir):
    
    min_num_nodes=100
    max_num_nodes=500
    
    adjs = []
    eigvals = []
    eigvecs = []
    n_nodes = []
    n_max = 0
    max_eigval = 0
    min_eigval = 0

    G = nx.Graph()
    # Load data
    path = os.path.join(data_dir, 'proteins/DD')
    data_adj = np.loadtxt(os.path.join(path, 'DD_A.txt'), delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(os.path.join(path, 'DD_graph_indicator.txt'), delimiter=',').astype(int)
    data_graph_types = np.loadtxt(os.path.join(path, 'DD_graph_labels.txt'), delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # Add edges
    G.add_edges_from(data_tuple)
    G.remove_nodes_from(list(nx.isolates(G)))

    # remove self-loop
    G.remove_edges_from(nx.selfloop_edges(G))

    # Split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1

    for i in tqdm(range(graph_num)):
        # Find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        G_sub.graph['label'] = data_graph_types[i]
        if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
            adj = torch.from_numpy(nx.adjacency_matrix(G_sub).toarray()).float()
            L = nx.normalized_laplacian_matrix(G_sub).toarray()
            L = torch.from_numpy(L).float()
            eigval, eigvec = torch.linalg.eigh(L)
            
            eigvals.append(eigval)
            eigvecs.append(eigvec)
            adjs.append(adj)
            n_nodes.append(G_sub.number_of_nodes())
            if G_sub.number_of_nodes() > n_max:
                n_max = G_sub.number_of_nodes()
            max_eigval = torch.max(eigval)
            if max_eigval > max_eigval:
                max_eigval = max_eigval
            min_eigval = torch.min(eigval)
            if min_eigval < min_eigval:
                min_eigval = min_eigval

    return adjs

def load_graphs(data_name, order='C-M'):
    raw_dir = f"resource/{data_name}"
    if data_name in ['GDSS_ego', 'GDSS_com', 'GDSS_enz', 'GDSS_grid']:
        with open(f'{raw_dir}.pkl', 'rb') as f:
            graphs = pickle.load(f)
    elif data_name == 'proteins':
        adjs = load_proteins_data(DATA_DIR)
        graphs = [adj_to_graph(adj.numpy()) for adj in adjs]
    else: # planar, sbm
        adjs, _, _, _, _, _, _, _ = torch.load(f'{raw_dir}.pt')
        graphs = [adj_to_graph(adj) for adj in adjs]
    
    train_graphs, val_graphs, test_graphs = train_val_test_split(graphs, data_name)
    
    graph_list = []
    for graphs in train_graphs, val_graphs, test_graphs:
        num_rep = 1
        # order graphs
        order_func = ORDER_FUNCS[order]
        total_ordered_graphs = order_graphs(graphs, num_repetitions=num_rep, order_func=order_func, is_mol=True)
        new_ordered_graphs = [nx.from_numpy_array(ordered_graph.to_adjacency().numpy()) for ordered_graph in tqdm(total_ordered_graphs, 'Map new ordered graphs')]
        graph_list.append(new_ordered_graphs)
    
    return graph_list

def get_max_len(data_name):
    graphs_list = load_graphs(data_name)
    max_len_edge = 0
    max_len_node = 0
    for graphs in graphs_list:
        max_edge = max([len(graph.edges) for graph in graphs])
        max_node = max([len(graph.nodes) for graph in graphs])
        print(max_node)
        if max_edge > max_len_edge:
            max_len_edge = max_edge
        if max_node > max_len_node:
            max_len_node = max_node
    return max_len_edge, max_len_node

def is_square(adj_flatten):
    num = len(adj_flatten)
    if(num >= 0):
        sr = int(math.sqrt(num))
        return ((sr*sr) == num)
    return False

def adj_flatten_to_adj(adj_flatten):
    matrix_size = int(math.sqrt(len(adj_flatten)))
    matrix = torch.tensor(adj_flatten)
    matrix = matrix.resize(matrix_size, matrix_size)
    return matrix

def is_symmetric(adj):
    return torch.all(adj.transpose(0,1) == adj).item()

def is_triangular(adj_flatten):
    num = len(adj_flatten)
    test_num = 1 + 8 * num
    sr = math.sqrt(test_num)
    return int(sr) ** 2 == test_num

def fill_lower_diag(array):
    # generate lower diagonal matrix with flatten sym
    n = int(np.sqrt(len(array) * 2)) + 1
    mask = np.tri(n, dtype=bool, k=-1)
    matrix = np.zeros((n, n), dtype=int)
    matrix[mask] = array
    return matrix

def fix_symmetry(adj):
    sym_adj = torch.tril(adj) + torch.tril(adj).T
    return torch.where(sym_adj>0, 1, 0)

def check_adj_list_validity(adj_list):
    # same multiple edges
    if len(adj_list) != len(set(adj_list)):
        return False
    # invalid (negative) node number
    elif len(adj_list) != len([target for src, target in adj_list if target >= 0]):
        return False
    # valid
    else:
        return True

def seq_rel_to_adj(seq_rel):
    adj_list = []
    cur_node_num = 0
    for element in seq_rel:
        if element != 0:
            adj_list.append((cur_node_num, cur_node_num-element))
        else:
            cur_node_num += 1
            continue
    if check_adj_list_validity(adj_list):
        return torch.tensor(adj_list_to_adj(adj_list))
    else:
        return ''


def map_samples_to_adjs(samples, string_type):
    
    filtered_samples = [sample for sample in samples if len(sample) > 0]
    # map adj_list_diff to adj_list
    if string_type == 'adj_list_diff':
        samples = [adj_list_diff_to_adj_list(adj_list) for adj_list in filtered_samples]
    # map adjacecny matrices from samples
    if string_type in ['adj_list', 'adj_list_diff']:
        adjs = [torch.tensor(adj_list_to_adj(adj_list)) for adj_list in filtered_samples]
    elif string_type == 'adj_flatten':
        adjs = [adj_flatten_to_adj(adj_flatten) for adj_flatten in filtered_samples if is_square(adj_flatten)]
        adjs = [adj for adj in adjs if is_symmetric(adj)]
    elif string_type == 'adj_flatten_sym':
        lower_adjs = [fill_lower_diag(adj_flatten) for adj_flatten in filtered_samples if is_triangular(adj_flatten)]
        adjs = [fix_symmetry(torch.tensor(adj)) for adj in lower_adjs]
    elif string_type == 'adj_seq':
        filtered_samples = [sample for sample in filtered_samples if sample[0] == 0]
        # num_nodes = [sample.count(0) for sample in filtered_samples]
        adjs = [seq_rel_to_adj(seq_rel) for seq_rel in filtered_samples if len(seq_rel_to_adj(seq_rel))>0]
    else:
        assert False, 'No string type'
        
    return adjs