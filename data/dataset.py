import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import os

from data.data_utils import adj_to_k2_tree, tree_to_dfs_string, tree_to_bfs_string, map_tree_pe
from data.load_data import get_community_small_data, get_ego_small_data, get_gdss_enzymes_data,  \
    get_gdss_grid_data, get_grid_small_data, get_caveman_data
from data.tokens import tokenize


DATA_DIR = "gcg/resource"

DATASETS = {  # name: (graphs, num_repetitions)
    "GDSS_com": (get_community_small_data, 1),
    "GDSS_ego": (get_ego_small_data, 1),
    "GDSS_grid": (get_gdss_grid_data, 1),
    "GDSS_enz": (get_gdss_enzymes_data, 1),
    "grid_small": (get_grid_small_data, 1),
    "Caveman": (get_caveman_data, 1)
}

class K2TreeListDataset(Dataset):
    def __init__(self, ordered_graphs: list):
        super().__init__()
        self.tree_leaf_lists = Parallel(n_jobs=8)(delayed(adj_to_k2_tree)(graph.to_adjacency()) 
                                                  for graph in tqdm(ordered_graphs, "Converting adj into tree"))
        self.tree_lists = [tree_leaf[0] for tree_leaf in self.tree_leaf_lists]
        self.leaf_lists = [tree_leaf[1] for tree_leaf in self.tree_leaf_lists]
        self.tree_leaf_tensors = [torch.stack(tree_leaf[0]+tree_leaf[1]) for tree_leaf in self.tree_leaf_lists]
    
    def __len__(self):
        return len(self.tree_lists)
    
    def __getitem__(self, idx: int):
        
        return self.tree_leaf_tensors[idx]
        
    def collate_fn(batch):
        data = []
        for tree_leaf in batch:
            data.append(tree_leaf)
        return pad_sequence(data, batch_first=True, padding_value=0)
    
class K2TreeDataset(Dataset):
    def __init__(self, ordered_graphs: list, string_type='dfs'):
        super().__init__()
        self.string_type = string_type
        self.trees = Parallel(n_jobs=8)(delayed(adj_to_k2_tree)(graph.to_adjacency(), return_tree=True) 
                                        for graph in tqdm(ordered_graphs, "Converting adj into tree"))
        if string_type == 'dfs':
            self.strings = Parallel(n_jobs=8)(delayed(tree_to_dfs_string)(tree) 
                                          for tree in tqdm(self.trees, "Converting tree into string"))
        elif string_type in ['bfs', 'group', 'bfs-deg', 'bfs-tri', 'bfs-deg-group']:
            self.strings = Parallel(n_jobs=8)(delayed(tree_to_bfs_string)(tree, string_type) 
                                          for tree in tqdm(self.trees, "Converting tree into string"))
        self.max_depth = max([tree.depth() for tree in self.trees])
        
    def __len__(self):
        return len(self.trees)
    
    def __getitem__(self, idx: int):
        return torch.LongTensor(tokenize(self.strings[idx], self.string_type))
    
    def collate_fn(batch):
        data = []
        for string in batch:
            data.append(string)

        return pad_sequence(data, batch_first=True, padding_value=0)
    
class K2StringTreeDataset(Dataset):
    def __init__(self, ordered_graphs: list, string_type='dfs'):
        super().__init__()
        self.string_type = string_type
        self.org_trees = Parallel(n_jobs=8)(delayed(adj_to_k2_tree)(graph.to_adjacency(), return_tree=True) 
                                        for graph in tqdm(ordered_graphs, "Converting adj into tree"))
        self.trees = Parallel(n_jobs=8)(delayed(map_tree_pe)(tree) for tree in tqdm(self.org_trees, "Map tree PE"))
        if string_type == 'dfs':
            self.strings = Parallel(n_jobs=8)(delayed(tree_to_dfs_string)(tree) 
                                          for tree in tqdm(self.trees, "Converting tree into string"))
        elif string_type in ['bfs', 'group', 'bfs-deg', 'bfs-tri', 'bfs-deg-group']:
            self.strings = Parallel(n_jobs=8)(delayed(tree_to_bfs_string)(tree, string_type) 
                                          for tree in tqdm(self.trees, "Converting tree into string"))
        self.max_depth = max([tree.depth() for tree in self.trees])
        
    def __len__(self):
        return len(self.org_trees)
    
    def __getitem__(self, idx: int):
        return torch.LongTensor(tokenize(self.strings[idx], self.string_type)), self.trees[idx]
    
    def collate_fn(batch):
        data = []
        tree_list = []
        for string, tree in batch:
            data.append(string)
            tree_list.append(tree)
        return pad_sequence(data, batch_first=True, padding_value=0), tree_list
    
class EgoDataset(Dataset):
    data_name = "ego_small"
    raw_dir = f"{DATA_DIR}/GDSS_ego"
    def __init__(self, split, string_type='bfs', is_tree=False):
        self.string_type = string_type
        self.is_tree = is_tree
        string_path = os.path.join(self.raw_dir, f"{self.data_name}_str_{split}.pkl")
        with open(string_path, 'rb') as f:
            self.strings = pickle.load(f)

        # tree_path = os.path.join(self.raw_dir, f"{self.data_name}_tree_{split}.pkl")
        # with open(tree_path, 'rb') as f:
        #     self.trees = pickle.load(f)
        # self.max_depth = max([tree.depth() for tree in self.trees])
        
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx: int):
        # if self.is_tree:
        #     return torch.LongTensor(tokenize(self.strings[idx], self.string_type)), self.trees[idx]
        # else:
        return torch.LongTensor(tokenize(self.strings[idx], self.string_type))
        
    # def collate_fn(batch):
    #     data = []
    #     tree_list = []
    #     for string, tree in batch:
    #         data.append(string)
    #         tree_list.append(tree)
    #     return pad_sequence(data, batch_first=True, padding_value=0), tree_list
    
class ComDataset(EgoDataset):
    data_name = 'community_small'
    raw_dir = f'{DATA_DIR}/GDSS_com'
    
class EnzDataset(EgoDataset):
    data_name = 'ENZYMES'
    raw_dir = f'{DATA_DIR}/GDSS_enz'

class GridDataset(EgoDataset):
    data_name = 'grid'
    raw_dir = f'{DATA_DIR}/GDSS_grid'
    
class GridSmallDataset(EgoDataset):
    data_name = 'grid_small'
    raw_dir = f'{DATA_DIR}/grid_small'


    # def collate_fn(batch):
    #     data = []
    #     for string in batch:
    #         data.append(string)
    #     return pad_sequence(data, batch_first=True, padding_value=0)