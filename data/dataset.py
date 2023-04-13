import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from joblib import Parallel, delayed
from tqdm import tqdm

from data.data_utils import adj_to_k2_tree, tree_to_dfs_string, tree_to_bfs_string, map_tree_pe
from data.load_data import get_community_small_data, get_ego_small_data, get_gdss_enzymes_data,  \
    get_gdss_grid_data, get_grid_small_data, get_caveman_data
from data.tokens import tokenize, TOKENS_BFS, TOKENS_DFS


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
        elif string_type in ['bfs', 'group', 'bfs-deg']:
            self.strings = Parallel(n_jobs=8)(delayed(tree_to_bfs_string)(tree, string_type) 
                                          for tree in tqdm(self.trees, "Converting tree into string"))
        self.max_depth = max([tree.depth() for tree in self.trees])
        
    def __len__(self):
        return len(self.trees)
    
    def __getitem__(self, idx: int):
        return torch.LongTensor(tokenize(self.strings[idx], self.string_type))
    
    def collate_fn(batch):
        data = []
        # tree_list = []
        for string in batch:
            data.append(string)
            # tree_list.append(tree)
        return pad_sequence(data, batch_first=True, padding_value=0)
    
class K2StringTreeDataset(Dataset):
    def __init__(self, ordered_graphs: list, string_type='dfs'):
        super().__init__()
        self.string_type = string_type
        trees = Parallel(n_jobs=8)(delayed(adj_to_k2_tree)(graph.to_adjacency(), return_tree=True) 
                                        for graph in tqdm(ordered_graphs, "Converting adj into tree"))
        self.trees = Parallel(n_jobs=8)(delayed(map_tree_pe)(tree) for tree in tqdm(trees, "Map tree PE"))
        if string_type == 'dfs':
            self.strings = Parallel(n_jobs=8)(delayed(tree_to_dfs_string)(tree) 
                                          for tree in tqdm(self.trees, "Converting tree into string"))
        elif string_type in ['bfs', 'group']:
            self.strings = Parallel(n_jobs=8)(delayed(tree_to_bfs_string)(tree) 
                                          for tree in tqdm(self.trees, "Converting tree into string"))
            
    def __len__(self):
        return len(self.trees)
    
    def __getitem__(self, idx: int):
        return torch.LongTensor(tokenize(self.strings[idx], self.string_type)), self.trees[idx]
    
    def collate_fn(batch):
        data = []
        tree_list = []
        for string, tree in batch:
            data.append(string)
            tree_list.append(tree)
        return pad_sequence(data, batch_first=True, padding_value=0), tree_list