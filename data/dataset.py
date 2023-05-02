import torch
from torch.utils.data import Dataset
from pathlib import Path
import os

from data.data_utils import map_deg_string, remove_redundant
from data.tokens import tokenize


DATA_DIR = "resource"
    
class EgoDataset(Dataset):
    data_name = "GDSS_ego"
    raw_dir = f"{DATA_DIR}/GDSS_ego"
    def __init__(self, split, string_type='bfs', order='C-M', is_tree=False):
        self.string_type = string_type
        self.is_tree = is_tree
        self.order = order
        string_path = os.path.join(self.raw_dir, f"{self.order}/{self.data_name}_str_{split}.txt")
        self.strings = Path(string_path).read_text(encoding="utf=8").splitlines()
        # use tree degree information
        if self.string_type in ['bfs-deg', 'bfs-deg-group']:
            self.strings = [map_deg_string(string) for string in self.strings]
        # remove redundant
        if 'red' in self.string_type:
            self.strings = [remove_redundant(string) for string in self.strings]
    
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx: int):
        return torch.LongTensor(tokenize(self.strings[idx], self.string_type))
    
class ComDataset(EgoDataset):
    data_name = 'GDSS_com'
    raw_dir = f'{DATA_DIR}/GDSS_com'
    
class EnzDataset(EgoDataset):
    data_name = 'GDSS_enz'
    raw_dir = f'{DATA_DIR}/GDSS_enz'

class GridDataset(EgoDataset):
    data_name = 'GDSS_grid'
    raw_dir = f'{DATA_DIR}/GDSS_grid'
    
class GridSmallDataset(EgoDataset):
    data_name = 'grid_small'
    raw_dir = f'{DATA_DIR}/grid_small'

class QM9Dataset(EgoDataset):
    data_name = "qm9"
    raw_dir = f"{DATA_DIR}/qm9"
        
class ZINCDataset(EgoDataset):
    data_name = 'zinc'
    raw_dir = f'{DATA_DIR}/zinc'
    
class PlanarDataset(EgoDataset):
    data_name = 'planar'
    raw_dir = f'{DATA_DIR}/planar'
            
class SBMDataset(EgoDataset):
    data_name = 'sbm'
    raw_dir = f'{DATA_DIR}/sbm'

class ProteinsDataset(EgoDataset):
    data_name = 'proteins'
    raw_dir = f'{DATA_DIR}/proteins'