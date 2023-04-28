import torch
from torch.utils.data import Dataset
from pathlib import Path
import pickle
import os
from collections import deque
import numpy as np
from itertools import compress, islice

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
        # use tree degree information
        if self.string_type in ['bfs-deg', 'bfs-deg-group']:
            self.strings = [self.map_deg_string(string) for string in self.strings]
        # remove redundant
        if 'red' in self.string_type:
            self.strings = [self.remove_redundant(string) for string in self.strings]
    
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
    
    def remove_redundant(self, input_string):
        string = input_string[0:4]
        pos_list = [1,2,3,4]
        str_pos_queue = deque([(s, p) for s, p in zip(string, pos_list)])
        for i in np.arange(4,len(input_string),4):
            cur_string = input_string[i:i+4]
            cur_parent, cur_parent_pos = str_pos_queue.popleft()
            # if value is 0, it cannot be parent node -> skip
            while((cur_parent == '0') and (len(str_pos_queue) > 0)):
                cur_parent, cur_parent_pos = str_pos_queue.popleft()
            # i: order of the child node in the same parent
            cur_pos = [cur_parent_pos*10+i for i in range(1,1+len(cur_string))]
            # pos_list: final position of each node
            pos_list.extend(cur_pos)
            str_pos_queue.extend([(s, c) for s, c in zip(cur_string, cur_pos)])
        
        pos_list = [str(pos) for pos in pos_list]
        # find positions ends with 2 including only 1 and 4
        remove_pos_prefix_list = [pos for i, pos in enumerate(pos_list) 
                                  if (pos[-1] == '2') and len((set(pos[:-1]))-set(['1', '4']))==0]
        remain_pos_index = [not pos.startswith(tuple(remove_pos_prefix_list)) for pos in pos_list]
        remain_pos_list = [pos for pos in pos_list if not pos.startswith(tuple(remove_pos_prefix_list))]
        # find cutting points (one block)
        cut_list = [i for i, pos in  enumerate(remain_pos_list) if pos[-1] == '4']
        cut_list_2 = [0]
        cut_list_2.extend(cut_list[:-1])
        cut_size_list = [i - j for i, j in zip(cut_list , cut_list_2)]
        cut_size_list[0] += 1
        
        final_string_list = list(compress([*input_string], remain_pos_index))

        pos_list_iter = iter(final_string_list)
        final_string_cut_list = [list(islice(pos_list_iter, i)) for i in cut_size_list]
        
        return [''.join(l) for l in final_string_cut_list]
    
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