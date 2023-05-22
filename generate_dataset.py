import torch
from sklearn.model_selection import train_test_split
import networkx as nx
import pickle
from tqdm import tqdm

from data.orderings import ORDER_FUNCS, order_graphs
from data.data_utils import generate_string
from data.mol_utils import generate_mol_string


DATA_DIR='gcg/resource'

dataset = ['GDSS_com', 'GDSS_ego', 'GDSS_enz', 'GDSS_grid', 'planar', 'sbm']
for data in dataset:
    generate_string(data, order='C-M')

# mol_dataset = ['qm9', 'zinc']
# for data in mol_dataset:
#     generate_mol_string(data)
