import torch
from torch.utils.data import Dataset
from pathlib import Path
import pickle
import os
from collections import deque

from data.data_utils import grouper
from data.load_data import get_community_small_data, get_ego_small_data, get_gdss_enzymes_data,  \
    get_gdss_grid_data, get_grid_small_data, get_caveman_data
from data.tokens import tokenize


DATA_DIR = "resource"

DATASETS = {  # name: (graphs, num_repetitions)
    "GDSS_com": (get_community_small_data, 1),
    "GDSS_ego": (get_ego_small_data, 1),
    "GDSS_grid": (get_gdss_grid_data, 1),
    "GDSS_enz": (get_gdss_enzymes_data, 1),
    "grid_small": (get_grid_small_data, 1),
    "Caveman": (get_caveman_data, 1)
}
    
class EgoDataset(Dataset):
    data_name = "ego_small"
    raw_dir = f"{DATA_DIR}/GDSS_ego"
    def __init__(self, split, string_type='bfs', is_tree=False):
        self.string_type = string_type
        self.is_tree = is_tree
        string_path = os.path.join(self.raw_dir, f"{self.data_name}_str_{split}.pkl")
        with open(string_path, 'rb') as f:
            self.strings = pickle.load(f)
        if self.string_type in ['bfs-deg', 'bfs-deg-group']:
            self.strings = [self.map_deg_string(string) for string in self.strings]
    
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx: int):
        return torch.LongTensor(tokenize(self.strings[idx], self.string_type))
    
    def map_deg_string(self, string):
        new_string = []
        group_queue = deque(grouper(4, string))
        group_queue.popleft()
        for index, char in enumerate(string):
            if len(group_queue) == 0:
                left = string[index:]
                break
            if char == '0':
                new_string.append(char)
            else:
                new_string.append(str(sum([int(char) for char in group_queue.popleft()])))
                
        return ''.join(new_string) + left
    
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

class QM9Dataset(EgoDataset):
    data_name = "qm9"
    raw_dir = f"{DATA_DIR}/qm9"
    def __init__(self, split, string_type='bfs', is_tree=False):
        self.string_type = string_type
        self.is_tree = is_tree
        string_path = os.path.join(self.raw_dir, f"{self.data_name}_str_{split}.txt")
        self.strings = Path(string_path).read_text(encoding="utf=8").splitlines()
        
class ZINCDataset(QM9Dataset):
    data_name = 'zinc'
    raw_dir = f'{DATA_DIR}/zinc'