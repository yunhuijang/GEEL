import torch
from sklearn.model_selection import train_test_split
import networkx as nx
import pickle
from tqdm import tqdm

from data.orderings import ORDER_FUNCS, order_graphs
from data.load_data import generate_string


DATA_DIR='resource'

dataset = ['GDSS_com']
for data in dataset:
    generate_string(data, order='C-M', k=3)
    # generate_string(data, order='C-M', k=4)

# mol_dataset = ['qm9', 'zinc']
# for data in mol_dataset:
#     generate_mol_string(data)
