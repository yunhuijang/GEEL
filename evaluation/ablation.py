import os
from pathlib import Path
import torch
import pickle
import networkx as nx
from statistics import mean
import pandas as pd

from data.data_utils import remove_redundant, adj_to_graph
from data.load_data import load_proteins_data


DATA_DIR = "../resource"

def compute_compression_rate(data_name, order, is_red=False):
    raw_dir = f"{DATA_DIR}/{data_name}/{order}"
    total_strings = []
    # load k2 tree string
    for split in ['train', 'val', 'test']:
        string_path = os.path.join(raw_dir, f"{data_name}_str_{split}.txt")
        strings = Path(string_path).read_text(encoding="utf=8").splitlines()
        if is_red:
            strings = [''.join(remove_redundant(string)) for string in strings]
        
        total_strings.extend(strings)

    
    # load data
    if data_name in ['planar', 'sbm']:
        adjs, _, _, _, _, _, _, _ = torch.load(f'{DATA_DIR}/{data_name}/{data_name}.pt')
    elif data_name == 'proteins':
        adjs = load_proteins_data("../resource")
    else:
        with open (f'{DATA_DIR}/{data_name}/{data_name}.pkl', 'rb') as f:
            graphs = pickle.load(f)
        adjs = [nx.adjacency_matrix(graph) for graph in graphs]
        
    n_nodes = [adj.shape[0]*adj.shape[1] for adj in adjs]
    len_strings = [len(string) for string in total_strings]
    compression_rates = [length / n_node for n_node, length in zip(n_nodes, len_strings)]
    
    return mean(compression_rates)


datas = ['proteins']
orders = ['BFS', 'DFS', 'C-M']
result_df = pd.DataFrame(columns = datas, index=orders)

for data in datas:
    print(data)
    for order in orders:
        result_df[data][order] = round(compute_compression_rate(data, order, True), 3)

print(result_df)
# result_df.to_csv('compression.csv')