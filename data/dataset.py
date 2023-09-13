import torch
from torch.utils.data import Dataset
import networkx as nx
from torch_geometric.datasets import MNISTSuperpixels

from data.data_utils import adj_to_adj_list
from data.tokens import tokenize
from data.orderings import bw_from_adj
from data.mol_tokens import tokenize_mol


DATA_DIR = "resource"
    
class EgoDataset(Dataset):
    data_name = "GDSS_ego"
    raw_dir = f"{DATA_DIR}/GDSS_ego"
    is_mol = False
    def __init__(self, graphs, string_type, is_token, vocab_size=200):
        self.adjs = [nx.adjacency_matrix(graph) for graph in graphs]
        self.adj_list = [adj_to_adj_list(adj) for adj in self.adjs]
        self.string_type = string_type
        self.bw = max([bw_from_adj(adj.toarray()) for adj in self.adjs])
        self.is_token = is_token
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.adj_list)
    
    def __getitem__(self, idx: int):
        return torch.LongTensor(tokenize(self.adjs[idx], self.adj_list[idx], self.data_name, self.string_type, self.is_token, self.vocab_size))
    
class ComDataset(EgoDataset):
    data_name = 'GDSS_com'
    raw_dir = f'{DATA_DIR}/GDSS_com'
    is_mol = False
    
class EnzDataset(EgoDataset):
    data_name = 'GDSS_enz'
    raw_dir = f'{DATA_DIR}/GDSS_enz'
    is_mol = False

class GridDataset(EgoDataset):
    data_name = 'GDSS_grid'
    raw_dir = f'{DATA_DIR}/GDSS_grid'
    is_mol = False
    
class GridSmallDataset(EgoDataset):
    data_name = 'grid_small'
    raw_dir = f'{DATA_DIR}/grid_small'
    is_mol = False

class PlanarDataset(EgoDataset):
    data_name = 'planar'
    raw_dir = f'{DATA_DIR}/planar'
    is_mol = False
            
class SBMDataset(EgoDataset):
    data_name = 'sbm'
    raw_dir = f'{DATA_DIR}/sbm'
    is_mol = False

class ProteinsDataset(EgoDataset):
    data_name = 'proteins'
    raw_dir = f'{DATA_DIR}/proteins'
    is_mol = False
    
class LobsterDataset(EgoDataset):
    data_name = 'lobster'
    raw_dir = f'{DATA_DIR}/lobster'
    is_mol = False
    
class PointCloudDataset(EgoDataset):
    data_name = 'point'
    raw_dir = f'{DATA_DIR}/point'
    is_mol = False
    
class EgoLargeDataset(EgoDataset):
    data_name = 'ego'
    raw_dir = f'{DATA_DIR}/ego'
    is_mol = False
    
class ComLargeDataset(EgoDataset):
    data_name = 'community'
    raw_dir = f'{DATA_DIR}/community'
    is_mol = False  

class QM9Dataset(EgoDataset):
    data_name = "qm9"
    raw_dir = f"{DATA_DIR}/qm9"
    is_mol = True
    
    def __init__(self, graphs, string_type, is_token, vocab_size=200):
        self.graphs = graphs
        self.adjs = [nx.adjacency_matrix(graph) for graph in graphs]
        self.adj_list = [adj_to_adj_list(adj) for adj in self.adjs]
        self.string_type = string_type
        self.bw = max([bw_from_adj(adj.toarray()) for adj in self.adjs])
        self.is_token = is_token
        self.vocab_size = vocab_size
        
    # for node, edge features
    def __getitem__(self, idx: int):
        if self.string_type in ['adj_seq_merge', 'adj_seq_rel_merge', 'adj_list', 'adj_list_diff', 'adj_list_diff_ni']:
            return torch.LongTensor(tokenize_mol(self.adjs[idx], self.adj_list[idx], nx.get_node_attributes(self.graphs[idx], 'x'), nx.get_edge_attributes(self.graphs[idx], 'edge_attr') , self.data_name, self.string_type))
        else:
        # return tokenize adj, tokenize_feature
            return (torch.LongTensor(tokenize(self.adjs[idx], self.adj_list[idx], self.data_name, self.string_type, self.is_token, self.vocab_size)), 
                torch.LongTensor(tokenize_mol(self.adjs[idx], self.adj_list[idx], nx.get_node_attributes(self.graphs[idx], 'x'), nx.get_edge_attributes(self.graphs[idx], 'edge_attr') , self.data_name, self.string_type)))
    
class ZINCDataset(QM9Dataset):
    data_name = 'zinc'
    raw_dir = f'{DATA_DIR}/zinc'
    is_mol = True
    
class MosesDataset(QM9Dataset):
    data_name = 'moses'
    raw_dir = f"{DATA_DIR}/moses"
    is_mol = True
    
class GuacamolDataset(QM9Dataset):
    data_name = 'guacamol'
    raw_dir = f"{DATA_DIR}/guacamol"
    is_mol = True
