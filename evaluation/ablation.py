import networkx as nx
# from scipy.stats import entropy
import numpy as np
import math
from math import log1p, log, log10
from itertools import chain
import pickle
import pandas as pd

from data.data_utils import adj_to_adj_list, load_graphs
from data.tokens import tokenize, token_to_id
# from data.load_data import load_proteins_data


DATA_DIR = "../resource"

# def compute_compression_rate(data_name, order, is_red=False):
#     raw_dir = f"{DATA_DIR}/{data_name}/{order}"
#     total_strings = []
#     # load k2 tree string
#     for split in ['train', 'val', 'test']:
#         string_path = os.path.join(raw_dir, f"{data_name}_str_{split}.txt")
#         strings = Path(string_path).read_text(encoding="utf=8").splitlines()
#         if is_red:
#             strings = [''.join(remove_redundant(string)) for string in strings]
        
#         total_strings.extend(strings)

    
#     # load data
#     if data_name in ['planar', 'sbm']:
#         adjs, _, _, _, _, _, _, _ = torch.load(f'{DATA_DIR}/{data_name}/{data_name}.pt')
#     # elif data_name == 'proteins':
#     #     adjs = load_proteins_data("../resource")
#     else:
#         with open (f'{DATA_DIR}/{data_name}/{data_name}.pkl', 'rb') as f:
#             graphs = pickle.load(f)
#         adjs = [nx.adjacency_matrix(graph) for graph in graphs]
        
#     n_nodes = [adj.shape[0]*adj.shape[1] for adj in adjs]
#     len_strings = [len(string) for string in total_strings]
#     compression_rates = [length / n_node for n_node, length in zip(n_nodes, len_strings)]
    
#     return mean(compression_rates)

def entropy(q, num_graphs):
    return -np.log(q)/num_graphs 

def compute_graph_entropy(tokens, counts, total_dict, num_graphs, num_tokens, type='product'):    
    if type == 'product':
        q = np.float128(1)
        for token, count in zip(tokens, counts):
            q *= (total_dict[token]**count)
        q *= num_graphs / (num_tokens + num_graphs)
        return entropy(q, num_graphs)
    elif type == 'sum':
        sum_log_q = 0
        for token, count in zip(tokens, counts):
            logq = -count * np.log(total_dict[token])
            sum_log_q += logq
        sum_log_q += -np.log(num_graphs / (num_tokens + num_graphs))
        return sum_log_q / num_graphs


def compute_total_entropy(data_name, string_type, type='product', is_token=False, split='train', order='C-M'):
    train_graphs, val_graphs, test_graphs = load_graphs(data_name, order)
    graphs_dict = {'train': train_graphs, 'test': test_graphs, 'val': val_graphs}
    adjs = [nx.adjacency_matrix(graph) for graph in graphs_dict[split]]
    num_graphs = len(adjs)
    adj_lists = [adj_to_adj_list(adj) for adj in adjs]
    tokenized_seqs = [tokenize(adj, adj_list, data_name, string_type, is_token) for adj, adj_list in zip(adjs, adj_lists)]
    # return max([len(seq) for seq in tokenized_seqs])
    tokens_list = [np.unique(tok_seq, return_counts=True)[0] for tok_seq in tokenized_seqs]
    counts_list = [np.unique(tok_seq, return_counts=True)[1] for tok_seq in tokenized_seqs]
    total_list = list(chain(*tokenized_seqs))
    total_tokens, total_counts = np.unique(total_list, return_counts=True)
    num_tokens = np.sum(total_counts)
    # num_tokens + num_grpahs: for end of token
    total_probs = total_counts / (num_tokens + num_graphs)
    total_dict = {token: prob for token, prob in zip(total_tokens, total_probs)}
    entropy_list = [compute_graph_entropy(tokens, counts, total_dict, num_graphs, num_tokens, type) 
                    for tokens, counts in zip(tokens_list, counts_list)]
    # math error -> replace with min_qs
    # min_qs = min(qs)
    # qs = [q if q != 1 else min_qs for q in qs]
    # entropy_list = [entropy(q, num_graphs) for q in qs]
    return np.mean(entropy_list)
    
dataset_list = ['GDSS_ego', 'GDSS_com', 'planar', 'GDSS_enz', 'sbm']
# dataset_list = ['GDSS_grid']
# string_type_list = ['adj_flatten', 'adj_flatten_sym', 'adj_list', 'adj_list_diff', 'adj_seq', 'adj_seq_rel']
string_type_list = ['adj_flatten', 'adj_flatten_sym']
# string_type_list = ['adj_flatten']
# string_type_list = ['adj_seq']
df = pd.DataFrame(index=string_type_list, columns=dataset_list)
for dataset in dataset_list:
# for dataset in ['GDSS_com']:
    print(dataset)
    for string_type in string_type_list:
    # for string_type in ['adj_list', 'adj_flatten', 'adj_flatten_sym', 'adj_seq']:
        print(string_type)
        ent = round(compute_total_entropy(dataset, string_type, type='sum', is_token=True), 3)
        print(ent)
        df.loc[string_type, dataset] = ent
df.to_csv('result_tok.csv')