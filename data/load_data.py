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


DATA_DIR = "resource"

def generate_string(dataset_name, order='C-M'):
    '''
    Generate strings for each dataset / split (without degree (only 0-1))
    '''
    # load molecule graphs
    if dataset_name in ['planar', 'sbm']:
        adjs, _, _, _, _, _, _, _ = torch.load(f'{DATA_DIR}/{dataset_name}/{dataset_name}.pt')
        graphs = [adj_to_graph(adj.numpy()) for adj in adjs]
        
    elif dataset_name == 'proteins':
        adjs = load_proteins_data(DATA_DIR)
        graphs = [adj_to_graph(adj.numpy()) for adj in adjs]
    else:
        with open (f'{DATA_DIR}/{dataset_name}/{dataset_name}.pkl', 'rb') as f:
            graphs = pickle.load(f)
    train_graphs, val_graphs, test_graphs = train_val_test_split(graphs)
    graph_list = []
    for graphs in train_graphs, val_graphs, test_graphs:
        num_rep = 1
        # order graphs
        order_func = ORDER_FUNCS[order]
        total_ordered_graphs = order_graphs(graphs, num_repetitions=num_rep, order_func=order_func, seed=0, is_mol=True)
        new_ordered_graphs = [map_new_ordered_graph(graph) for graph in tqdm(total_ordered_graphs, 'Map new ordered graphs')]
        graph_list.append(new_ordered_graphs)
    
    # write graphs
    splits = ['train', 'val', 'test']
    
    for graphs, split in zip(graph_list, splits):
        adjs = [nx.adjacency_matrix(graph, range(len(graph))) for graph in graphs]
        trees = [adj_to_k2_tree(torch.Tensor(adj.todense()), return_tree=True, is_mol=False) for adj in tqdm(adjs, 'Generating tree from adj')]
        strings = [tree_to_bfs_string(tree, string_type='group') for tree in tqdm(trees, 'Generating strings from tree')]
        file_name = f'{dataset_name}_str_{split}'
        with open(f'{DATA_DIR}/{dataset_name}/{order}/{file_name}.txt', 'w') as f:
            for string in strings:
                f.write(f'{string}\n')
        if split == 'test':
            with open(f'{DATA_DIR}/{dataset_name}/{order}/{dataset_name}_test_graphs.pkl', 'wb') as f:
                pickle.dump(graphs, f)
                
def generate_mol_string(dataset_name, order='C-M', is_small=False):
    '''
    Generate strings for each dataset / split (without degree (only 0-1))
    '''
    # load molecule graphs
    col_dict = {'qm9': 'SMILES1', 'zinc': 'smiles'}
    df = pd.read_csv(f'{DATA_DIR}/{dataset_name}/{dataset_name}.csv')
    smiles = list(df[col_dict[dataset_name]])
    if is_small:
        smiles = smiles[:100]
    smiles = [s for s in smiles if len(s)>1]
    smiles = canonicalize_smiles(smiles)
    splits = ['train', 'val', 'test']
    train_smiles, val_smiles, test_smiles = train_val_test_split(smiles)
    for s, split in zip([train_smiles, val_smiles, test_smiles], splits):
        with open(f'{DATA_DIR}/{dataset_name}/{dataset_name}_smiles_{split}.txt', 'w') as f:
            for string in s:
                f.write(f'{string}\n')
    graph_list = []
    for smiles in train_smiles, val_smiles, test_smiles:
        mols = smiles_to_mols(smiles)
        graphs = mols_to_nx(mols)
        graphs = [add_self_loop(graph) for graph in tqdm(graphs, 'Adding self-loops')]
        num_rep = 1
        # order graphs
        order_func = ORDER_FUNCS[order]
        total_graphs = graphs
        total_ordered_graphs = order_graphs(total_graphs, num_repetitions=num_rep, order_func=order_func, seed=0, is_mol=True)
        new_ordered_graphs = [map_new_ordered_graph(graph) for graph in tqdm(total_ordered_graphs, 'Map new ordered graphs')]
        graph_list.append(new_ordered_graphs)
    
    # write graphs
    
    for graphs, split in zip(graph_list, splits):
        weighted_adjs = [nx.attr_matrix(graph, edge_attr='label', rc_order=range(len(graph))) for graph in graphs]
        trees = [adj_to_k2_tree(torch.Tensor(adj), return_tree=True, is_mol=True) for adj in tqdm(weighted_adjs, 'Generating tree from adj')]
        strings = [tree_to_bfs_string_mol(tree, string_type='group') for tree in tqdm(trees, 'Generating strings from tree')]
        if is_small:
            file_name = f'{dataset_name}_small_str_{split}'
        else:
            file_name = f'{dataset_name}_str_{split}'
        with open(f'{DATA_DIR}/{dataset_name}/{file_name}.txt', 'w') as f:
            for string in strings:
                f.write(f'{string}\n')
        if split == 'test':
            with open(f'{DATA_DIR}/{dataset_name}/{dataset_name}_test_graphs.pkl', 'wb') as f:
                pickle.dump(graphs, f)
                
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