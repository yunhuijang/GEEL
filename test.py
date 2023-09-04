import math
import torch
import numpy as np
import networkx as nx
import sentencepiece as spm
import pickle

from data.orderings import bw_from_adj
from data.data_utils import unflatten_forward, train_data_to_string, get_max_len, generate_vocabulary
from data.mol_utils import canonicalize_smiles, mols_to_nx, smiles_to_mols


from torch_geometric.datasets import MNISTSuperpixels
import moses

for dataset in ['ego', 'proteins', 'lobster']:
    for string_type in ['adj_seq_rel_blank', 'adj_seq_blank']:
        vocab_size = 70
        generate_vocabulary(dataset, string_type, vocab_size)
        
for dataset in ['GDSS_grid']:
    for string_type in ['adj_seq_rel_blank', 'adj_seq_blank']:
        vocab_size = 65
        generate_vocabulary(dataset, string_type, vocab_size)