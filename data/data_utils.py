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
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import to_networkx
from scipy.sparse import lil_matrix, vstack
import sentencepiece as spm

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

def featured_adj_list_to_adj(adj_list):
    '''
    edge featured adjacency list to weighted adjacency matrix
    '''
    if len(adj_list) < 2:
        num_nodes = len(adj_list)
        adj = [[0] * num_nodes for _ in range(num_nodes)]
        return np.array(adj)
    
    num_nodes =  max(map(max, adj_list))+1
    adj = [[0] * num_nodes for _ in range(num_nodes)]
    
    for n, e, f in adj_list:
        adj[n][e] = f
        adj[e][n] = f

    return np.array(adj)

def adj_list_diff_to_adj_list(adj_list_diff):
    return [(token[0], token[0]-token[1]) for token in adj_list_diff]

    
def train_val_test_split(
    data: list, data_name='GDSS_com',
    train_size: float = 0.7, val_size: float = 0.1, seed: int = 42,
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
    elif data_name in ['point', 'lobster']:
        # npr = np.random.RandomState(seed)
        # npr.shuffle(data)
        val_size = 0.2
        train = data[int(val_size*len(data)):int((train_size+val_size)*len(data))]
        val = data[:int(val_size*len(data))]
        test = data[int((train_size+val_size)*len(data)):]
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

def load_point_data(data_dir, min_num_nodes, max_num_nodes, node_attributes, graph_labels):
    print('Loading point cloud dataset')
    name = 'FIRSTMM_DB'
    G = nx.Graph()
    # load data
    path = os.path.join(data_dir, name)
    data_adj = np.loadtxt(
        os.path.join(path, f'{name}_A.txt'), delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(os.path.join(path, f'{name}_node_attributes.txt'), 
                                   delimiter=',')
    data_node_label = np.loadtxt(os.path.join(path, f'{name}_node_labels.txt'), 
                                 delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(os.path.join(path, f'{name}_graph_indicator.txt'),
                                      delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(os.path.join(path, f'{name}_graph_labels.txt'), 
                                       delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
            G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # remove self-loop
    G.remove_edges_from(nx.selfloop_edges(G))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]

        if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
            graphs.append(G_sub)
        if G_sub.number_of_nodes() > max_nodes:
            max_nodes = G_sub.number_of_nodes()
            
    print('Loaded')
    return graphs

# Codes adpated from https://github.com/JiaxuanYou/graph-generation
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_ego_data(dataset):
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pickle.load(open(f"{DATA_DIR}/ego/ind.{dataset}.{names[i]}", 'rb'), encoding='latin1')
        objects.append(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"{DATA_DIR}/ego/ind.{dataset}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    tx_extended = lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    tx = tx_extended

    features = vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G

def n_community(c_sizes, p_inter=0.05):
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.3, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = [G.subgraph(c) for c in nx.connected_components(G)]
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i+1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    return G

def load_graphs(data_name, order='C-M'):
    raw_dir = f"resource/{data_name}"
    if data_name in ['GDSS_ego', 'GDSS_com', 'GDSS_enz', 'GDSS_grid']:
        with open(f'{raw_dir}.pkl', 'rb') as f:
            graphs = pickle.load(f)
    elif data_name == 'proteins':
        adjs = load_proteins_data(DATA_DIR)
        graphs = [adj_to_graph(adj.numpy()) for adj in adjs]
    elif data_name == 'mnist':
        train_graphs, val_graphs, test_graphs = mnist_to_graphs()
    # Codes adpadted from https://github.com/lrjconan/GRAN
    elif data_name == 'lobster':
        graphs = []
        p1 = 0.7
        p2 = 0.7
        count = 0
        min_node = 10
        max_node = 100
        max_edge = 0
        mean_node = 80
        num_graphs = 100

        seed_tmp = 1234
        while count < num_graphs:
            G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
            if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
                graphs.append(G)
                if G.number_of_edges() > max_edge:
                    max_edge = G.number_of_edges()
                count += 1
            seed_tmp += 1
    elif data_name == 'point':
        graphs = load_point_data(DATA_DIR, min_num_nodes=0, max_num_nodes=10000, 
                                  node_attributes=False, graph_labels=True)
    # Codes adpated from https://github.com/JiaxuanYou/graph-generation
    elif data_name == 'ego':
        _, _, G = load_ego_data(dataset='citeseer')
        G = max([G.subgraph(c) for c in nx.connected_components(G)], key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=3)
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                graphs.append(G_ego)
    elif data_name == 'community':
        graphs = []
        num_communities = 2
        print('Creating dataset with ', num_communities, ' communities')
        for k in range(500):
            np.random.seed(1234+k)
            c_sizes = np.random.choice(np.arange(30, 81), num_communities)
            graphs.append(n_community(c_sizes, p_inter=0.05))
    elif data_name in ['qm9', 'zinc']:
        graphs_list = []
        for split in ['train', 'val', 'test']:
            with open(f'resource/{data_name}/{data_name}_graph_{split}.pkl', 'rb') as f:
                graphs = pickle.load(f)
                graphs_list.append(graphs)
        train_graphs, val_graphs, test_graphs = graphs_list
    else: # planar, sbm
        adjs, _, _, _, _, _, _, _ = torch.load(f'{raw_dir}.pt')
        graphs = [adj_to_graph(adj) for adj in adjs]
        
    if data_name not in  ['mnist', 'qm9', 'zinc']:
        train_graphs, val_graphs, test_graphs = train_val_test_split(graphs, data_name)
    
    graph_list = []
    for graphs in train_graphs, val_graphs, test_graphs:
        num_rep = 1
        # order graphs
        order_func = ORDER_FUNCS[order]
        
        if data_name == 'mnist':
            total_ordered_graphs = order_graphs(graphs, num_repetitions=num_rep, order_func=order_func, is_mol=True)
            new_ordered_graphs = [to_networkx(ordered_graph.to_mnist_data()) for ordered_graph in tqdm(total_ordered_graphs, 'Map new ordered graphs')]
        elif data_name in ['qm9', 'zinc']:
            total_ordered_graphs = order_graphs(graphs, num_repetitions=num_rep, order_func=order_func, is_mol=True)
            new_ordered_graphs = [to_networkx(ordered_graph.to_mol_data(), node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True) for ordered_graph in tqdm(total_ordered_graphs, 'Map new ordered graphs')]
        else:
            total_ordered_graphs = order_graphs(graphs, num_repetitions=num_rep, order_func=order_func, is_mol=False)
            new_ordered_graphs = [nx.from_numpy_array(ordered_graph.to_adjacency().numpy()) for ordered_graph in tqdm(total_ordered_graphs, 'Map new ordered graphs')]
        graph_list.append(new_ordered_graphs)
    
    return graph_list

def mnist_to_graphs():
    train_val_graphs = MNISTSuperpixels(root='resource', train=True)[:80]
    val_raw_data = train_val_graphs[:int(len(train_val_graphs)*0.2)]
    train_raw_data = train_val_graphs[int(len(train_val_graphs)*0.2):]
    test_raw_data = MNISTSuperpixels(root='resource', train=False)[:20]
    
    graphs = []
    for raw_data in [train_raw_data, val_raw_data, test_raw_data]:
        node_attrs = ['x', 'pos']
        graph_attrs = ['y']
        networkx_graphs = [to_networkx(data, node_attrs=node_attrs, graph_attrs=graph_attrs,
                                       to_undirected=True) for data in raw_data]
        graphs.append(networkx_graphs)
        
    return graphs

def get_max_len(data_name):
    graphs_list = load_graphs(data_name)
    max_len_edge = 0
    max_len_node = 0
    for graphs in graphs_list:
        max_edge = max([len(graph.edges) for graph in graphs])
        max_node = max([len(graph.nodes) for graph in graphs])
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

def seq_to_adj_list(seq):
    adj_list = []
    cur_node_num = 0
    for element in seq:
        if element != 0:
            adj_list.append((cur_node_num, cur_node_num-element))
        else:
            cur_node_num += 1
            continue
    return adj_list

def seq_rel_to_adj_list(seq_rel):
    adj_list = []
    cur_node_num = 0
    tar_node = 1
    for element in seq_rel:
        if element != 0:
            tar_node -= element
            adj_list.append((cur_node_num, tar_node))
        else:
            cur_node_num += 1
            tar_node = cur_node_num
            continue
    return adj_list

def seq_to_adj(seq):
    adj_list = seq_to_adj_list(seq)
    if check_adj_list_validity(adj_list):
        return torch.tensor(adj_list_to_adj(adj_list))
    else:
        return ''

def seq_rel_to_adj(seq):
    adj_list = seq_rel_to_adj_list(seq)
    if check_adj_list_validity(adj_list):
        return torch.tensor(adj_list_to_adj(adj_list))
    else:
        return ''

def map_samples_to_adjs(samples, string_type, is_token):
    
    filtered_samples = [sample for sample in samples if (len(sample) > 0) and ('<unk>' not in sample)]
    if is_token:
        filtered_samples = [''.join(sample) for sample in filtered_samples]
        filtered_samples = [sample.replace('▁', '').replace('<s>', '').replace('</s>', '') for sample in filtered_samples]
        filtered_samples = [[int(char) for char in sample] for sample in filtered_samples]
        filtered_samples = [sample for sample in filtered_samples if (len(sample) > 0)]

    # map adj_list_diff to adj_list
    if string_type == 'adj_list_diff':
        filtered_samples = [adj_list_diff_to_adj_list(adj_list) for adj_list in filtered_samples]
    # map adjacecny matrices from samples
    if string_type in ['adj_list', 'adj_list_diff']:
        adjs = [torch.tensor(adj_list_to_adj(adj_list)) for adj_list in filtered_samples if check_adj_list_validity(adj_list)>0]
    elif string_type == 'adj_flatten':
        adjs = [adj_flatten_to_adj(adj_flatten) for adj_flatten in filtered_samples if is_square(adj_flatten)]
        adjs = [adj for adj in adjs if is_symmetric(adj)]
    elif string_type == 'adj_flatten_sym':
        lower_adjs = [fill_lower_diag(adj_flatten) for adj_flatten in filtered_samples if is_triangular(adj_flatten)]
        adjs = [fix_symmetry(torch.tensor(adj)) for adj in lower_adjs]
    elif string_type == 'adj_seq':
        filtered_samples = [sample for sample in filtered_samples if sample[0] == 0]
        adjs = [seq_to_adj(seq_rel) for seq_rel in filtered_samples if len(seq_to_adj(seq_rel))>0]
    elif string_type == 'adj_seq_rel':
        filtered_samples = [sample for sample in filtered_samples if sample[0] == 0]
        adjs = [seq_rel_to_adj(seq_rel) for seq_rel in filtered_samples if len(seq_rel_to_adj(seq_rel))>0]
    elif string_type == 'bwr':
        adjs = [unflatten_forward(torch.tensor(flatten)) for flatten in filtered_samples]
    else:
        assert False, 'No string type'
        
    return adjs

# Codes adapted from https://github.com/Genentech/bandwidth-graph-generation
def flatten_forward(A: torch.Tensor, bw: int) -> torch.Tensor:
    n = A.shape[0]
    out = torch.zeros((n, bw) + A.shape[2:], dtype=A.dtype, device=A.device)
    for i in range(n):
        append_len = min(bw, n - i - 1)
        if append_len > 0:
            out[i, :append_len] = A[i, i + 1: i + 1 + append_len]
    return out

def unflatten_forward(band_flat_A: torch.Tensor) -> torch.Tensor:
    n, bw = band_flat_A.shape[:2]
    out = torch.zeros((n, n) + band_flat_A.shape[2:], dtype=band_flat_A.dtype, device=band_flat_A.device)
    for i in range(n):
        append_len = min(bw, n - i - 1)
        if append_len > 0:
            out[i, i + 1: i + 1 + append_len] = band_flat_A[i, :append_len]
    out = out + out.T
    return out

def unflatten_forward(band_flat_A: torch.Tensor) -> torch.Tensor:
    n, bw = band_flat_A.shape[:2]
    out = torch.zeros((n, n) + band_flat_A.shape[2:], dtype=band_flat_A.dtype, device=band_flat_A.device)
    for i in range(n):
        append_len = min(bw, n - i - 1)
        if append_len > 0:
            out[i, i + 1: i + 1 + append_len] = band_flat_A[i, :append_len]
    out = out + out.T
    return out

def map_string_adj_seq_rel(adj_list):
    string = "0"
    prev_src_node = 1
    adj_list = sorted(adj_list, key = lambda x: (x[0], -x[1]))
    cur_tar_node = adj_list[0][1]
    for src_node, tar_node in adj_list:
        if prev_src_node != src_node:
            string += "0"
            diff = src_node - tar_node
        else:
            diff = cur_tar_node - tar_node
        string += str(diff)
        prev_src_node = src_node
        cur_tar_node = tar_node
    return string

def map_string_adj_seq(adj_list):
    string = "0"
    prev_src_node = 1
    for src_node, tar_node in adj_list:
        if prev_src_node != src_node:
            string += "0"
        diff = src_node - tar_node
        string += str(diff)
        prev_src_node = src_node
    return string

def map_string_flat_sym(adj):
    np_adj = adj.toarray()
    lower_diagonal = np_adj[np.tril_indices(len(np_adj))]
    return "".join([str(int(elem)) for elem in lower_diagonal.tolist()])

def train_data_to_string(data_name='GDSS_com', string_type='adj_seq_rel', order='C-M'):
    '''
    Generate string representation for tokenization
    '''
    graphs, _, _ = load_graphs(data_name, order)
    adjs = [nx.adjacency_matrix(graph) for graph in graphs]
    adj_lists = [adj_to_adj_list(adj) for adj in adjs]

    if string_type == 'adj_seq_rel':
        strings = [map_string_adj_seq_rel(adj_list) for adj_list in adj_lists]

    elif string_type == 'adj_seq':
        strings = [map_string_adj_seq(adj_list) for adj_list in adj_lists]
        
    elif string_type == 'adj_flatten':
        strings = ["".join([str(int(elem)) for elem in torch.flatten(torch.tensor(adj.todense())).tolist()]) for adj in adjs]
        
    elif string_type == 'adj_flatten_sym':
        strings = [map_string_flat_sym(adj) for adj in adjs]
    print(max([len(string) for string in strings]))
    with open(f'./samples/string/{data_name}/{string_type}.txt', 'w') as f :
        for string in strings:
            f.write("%s\n" %string)
    
def generate_vocabulary(dataset_name, string_type, vocab_size):
    train_data_to_string(dataset_name, string_type)
    spm.SentencePieceTrainer.Train(f"--input=samples/string/{dataset_name}/{string_type}.txt --model_prefix=resource/tokenizer/{dataset_name}/{string_type}_{vocab_size} --vocab_size={vocab_size} --model_type=bpe --character_coverage=1.0 --max_sentence_length=160000 --input_sentence_size=10000000")