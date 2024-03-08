import torch
from torch.utils.data import Dataset
import networkx as nx
from torch_geometric.datasets import MNISTSuperpixels

from data.data_utils import adj_to_adj_list
from data.tokens import tokenize
from data.orderings import bw_from_adj
from data.mol_tokens import tokenize_mol


DATA_DIR = "resource"
    
class GeneralDataset(Dataset):
    data_name = "GDSS_ego"
    raw_dir = f"{DATA_DIR}/GDSS_ego"
    is_mol = False
    def __init__(self, graphs, string_type, is_token, vocab_size=200, order='C-M'):
        self.adjs = [nx.adjacency_matrix(graph) for graph in graphs]
        self.adj_list = [adj_to_adj_list(adj) for adj in self.adjs]
        self.string_type = string_type
        self.bw = max([bw_from_adj(adj.toarray()) for adj in self.adjs])
        self.is_token = is_token
        self.vocab_size = vocab_size
        self.order = order

    def __len__(self):
        return len(self.adj_list)
    
    def __getitem__(self, idx: int):
        return torch.LongTensor(tokenize(self.adjs[idx], self.adj_list[idx], self.data_name, self.string_type, self.is_token, self.vocab_size, self.order))

class EgoDataset(Dataset):
    data_name = "GDSS_ego"
    raw_dir = f"{DATA_DIR}/GDSS_ego"

class ComDataset(GeneralDataset):
    data_name = 'GDSS_com'
    raw_dir = f'{DATA_DIR}/GDSS_com'
    
class EnzDataset(GeneralDataset):
    data_name = 'GDSS_enz'
    raw_dir = f'{DATA_DIR}/GDSS_enz'

class GridDataset(GeneralDataset):
    data_name = 'GDSS_grid'
    raw_dir = f'{DATA_DIR}/GDSS_grid'
    
class GridSmallDataset(GeneralDataset):
    data_name = 'grid_small'
    raw_dir = f'{DATA_DIR}/grid_small'

class PlanarDataset(GeneralDataset):
    data_name = 'planar'
    raw_dir = f'{DATA_DIR}/planar'
            
class SBMDataset(GeneralDataset):
    data_name = 'sbm'
    raw_dir = f'{DATA_DIR}/sbm'

class ProteinsDataset(GeneralDataset):
    data_name = 'proteins'
    raw_dir = f'{DATA_DIR}/proteins'
    
class LobsterDataset(GeneralDataset):
    data_name = 'lobster'
    raw_dir = f'{DATA_DIR}/lobster'

    
class PointCloudDataset(GeneralDataset):
    data_name = 'point'
    raw_dir = f'{DATA_DIR}/point'
    
class EgoLargeDataset(GeneralDataset):
    data_name = 'ego'
    raw_dir = f'{DATA_DIR}/ego'

class Grid500Dataset(GeneralDataset):
    data_name = 'grid-500'
    raw_dir = f'{DATA_DIR}/grid-500'
    
class Grid1000Dataset(GeneralDataset):
    data_name = 'grid-1000'
    raw_dir = f'{DATA_DIR}/grid-1000'
    
class Grid2000Dataset(GeneralDataset):
    data_name = 'grid-2000'
    raw_dir = f'{DATA_DIR}/grid-2000'
    
class Grid5000Dataset(GeneralDataset):
    data_name = 'grid-5000'
    raw_dir = f'{DATA_DIR}/grid-5000'
    
class Grid10000Dataset(GeneralDataset):
    data_name = 'grid-10000'
    raw_dir = f'{DATA_DIR}/grid-10000'
    
class Grid20000Dataset(GeneralDataset):
    data_name = 'grid-20000'
    raw_dir = f'{DATA_DIR}/grid-20000'

class MoleculeDataset(Dataset):
    is_mol = True
    
    def __init__(self, graphs, string_type, is_token, vocab_size=200, order='C-M'):
        self.graphs = graphs
        self.adjs = [nx.adjacency_matrix(graph) for graph in graphs]
        self.adj_list = [adj_to_adj_list(adj) for adj in self.adjs]
        self.string_type = string_type
        self.bw = max([bw_from_adj(adj.toarray()) for adj in self.adjs])
        self.is_token = is_token
        self.vocab_size = vocab_size
        self.order = order
    
    def __len__(self):
        return len(self.adj_list)
    
    # for node, edge features
    def __getitem__(self, idx: int):
        return torch.LongTensor(tokenize_mol(self.adjs[idx], self.adj_list[idx], nx.get_node_attributes(self.graphs[idx], 'x'), nx.get_edge_attributes(self.graphs[idx], 'edge_attr') , self.data_name, self.string_type))

class QM9Dataset(MoleculeDataset):
    data_name = "qm9"
    raw_dir = f"{DATA_DIR}/qm9"

class ZINCDataset(MoleculeDataset):
    data_name = 'zinc'
    raw_dir = f'{DATA_DIR}/zinc'
    
class MosesDataset(MoleculeDataset):
    data_name = 'moses'
    raw_dir = f"{DATA_DIR}/moses"
    
class GuacamolDataset(MoleculeDataset):
    data_name = 'guacamol'
    raw_dir = f"{DATA_DIR}/guacamol"
