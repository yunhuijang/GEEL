import math
import torch
import numpy as np
import networkx as nx
import sentencepiece as spm
import pickle

from data.orderings import bw_from_adj
from data.data_utils import unflatten_forward, train_data_to_string, get_max_len
from data.mol_utils import canonicalize_smiles, mols_to_nx, smiles_to_mols

from torch_geometric.datasets import MNISTSuperpixels
import moses

# print(get_max_len('zinc'))

# train_data_to_string('qm9', 'adj_seq')
test_scaffolds = moses.get_dataset('test_scaffolds')

# for data in ['qm9', 'zinc']:
#     for split in ['train', 'test', 'val']:
        

#         with open(f'resource/{data}/{data}' + f'_smiles_{split}.txt', 'r') as f:
#             smiles = f.readlines()
#             smiles = canonicalize_smiles(smiles)
        
#         mols = smiles_to_mols(smiles)
#         graphs = mols_to_nx(mols)
        
#         with open(f'resource/{data}/{data}_graph_{split}.pkl', 'wb') as f:
#             pickle.dump(graphs, f)