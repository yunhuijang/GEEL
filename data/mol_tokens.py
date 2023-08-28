import torch
import numpy as np
from itertools import product
import sentencepiece as spm
from collections import defaultdict
from itertools import product

from data.data_utils import flatten_forward, map_string_adj_seq, map_string_adj_seq_rel, map_string_flat_sym
from data.orderings import bw_from_adj


PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
UNK_TOKEN = "<unk>"

standard_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

TOKENS_DICT_MOL = {}
TOKENS_DICT_FLATTEN_MOL = {}
TOKENS_DICT_SEQ_MOL = {}

NODE_TOKENS_DICT = {'qm9': ['F', 'O', 'N', 'C'], 'zinc': ['F', 'O', 'N', 'C', 'P', 'I', 'Cl', 'Br', 'S']}
bond_tokens = [5,6,7,8]
NODE_TYPE_DICT = {'F': 9, 'O': 10, 'N': 11, 'C': 12, 'P': 13, 'I': 14, 'Cl': 15, 'Br': 16, 'S': 17}
TYPE_NODE_DICT = {str(key): value for value, key in NODE_TYPE_DICT.items()}
BOND_TYPE_DICT = {1: 5, 2: 6, 3: 7, 1.5: 8}
TYPE_BOND_DICT = {key: value for value, key in NODE_TYPE_DICT.items()}

for dataset in ['qm9', 'zinc']:
    tokens = standard_tokens.copy()
    tokens.extend(NODE_TYPE_DICT[node_type] for node_type in NODE_TOKENS_DICT[dataset])
    tokens.extend([(src_bond, tar_bond) for src_bond, tar_bond in product(bond_tokens, bond_tokens)])
    TOKENS_DICT_MOL[dataset] = tokens
    
    tokens_seq = standard_tokens.copy()
    tokens_seq.extend(NODE_TYPE_DICT[node_type] for node_type in NODE_TOKENS_DICT[dataset])
    tokens_seq.extend(bond_tokens)
    TOKENS_DICT_SEQ_MOL[dataset] = tokens_seq
    
    tokens_flat = standard_tokens.copy()
    tokens_flat.extend(NODE_TYPE_DICT[node_type] for node_type in NODE_TOKENS_DICT[dataset])
    tokens_flat.extend(bond_tokens)
    # element of adjacency matrix 0
    tokens_flat.append(0)
    TOKENS_DICT_FLATTEN_MOL[dataset] = tokens_flat

def token_list_to_dict(tokens):
    return {token: i for i, token in enumerate(tokens)}

TOKENS_KEY_DICT_MOL = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_MOL.items()}
TOKENS_KEY_DICT_FLATTEN_MOL = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_FLATTEN_MOL.items()}
TOKENS_KEY_DICT_SEQ_MOL = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_SEQ_MOL.items()}

def token_to_id_mol(data_name, string_type):
    if string_type == ['adj_list', 'adj_list_diff']:
        return TOKENS_KEY_DICT_MOL[data_name]
    elif string_type in ['adj_flatten', 'adj_flatten_sym', 'bwr']:
        return TOKENS_KEY_DICT_FLATTEN_MOL[data_name]
    elif string_type in ['adj_seq', 'adj_seq_rel']:
        return TOKENS_KEY_DICT_SEQ_MOL[data_name]

def id_to_token_mol(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def tokenize_mol(adj, adj_list, node_attr, edge_attr, data_name, string_type):
    TOKEN2ID = token_to_id_mol(data_name, string_type)
    tokens = ["[bos]"]
    if string_type in ['adj_list', 'adj_list_diff']:
        tokens.extend(adj_list)
    elif string_type == 'adj_flatten':
        tokens.extend(torch.flatten(torch.tensor(adj.todense())).tolist())
    elif string_type == 'adj_flatten_sym':
        np_adj = adj.toarray()
        lower_diagonal = np_adj[np.tril_indices(len(np_adj))]
        tokens.extend(lower_diagonal.tolist())
    elif string_type in ['adj_seq', 'adj_seq_rel']:
        # longer than tokenize (because of the first node feature)
        tokens.append(node_attr[0])
        prev_src_node = 0
        for src_node, tar_node in adj_list:
            if prev_src_node != src_node:
                tokens.append(node_attr[src_node])
            tokens.append(edge_attr[(tar_node, src_node)])
            prev_src_node = src_node
    elif string_type == 'bwr':
        bw = bw_from_adj(adj.toarray())
        tokens.extend(torch.flatten(flatten_forward(torch.tensor(adj.todense()), bw)).tolist())
        
    tokens.append("[eos]")

    return [TOKEN2ID[token] for token in tokens]


def untokenize_mol(sequence, data_name, string_type, is_token, vocab_size=200):
    if string_type == 'adj_list':
        tokens = TOKENS_DICT_MOL[data_name]
    elif string_type == 'adj_list_diff':
        tokens = TOKENS_DICT_DIFF_MOL[data_name]
    elif string_type in ['adj_flatten', 'adj_flatten_sym', 'bwr']:
        tokens = TOKENS_DICT_FLATTEN_MOL[data_name]
    elif string_type in ['adj_seq', 'adj_seq_rel']:
        tokens = TOKENS_DICT_SEQ_MOL[data_name]
        
    ID2TOKEN = id_to_token_mol(tokens)
    tokens = [ID2TOKEN[id_] for id_ in sequence]

    org_tokens = tokens
    if tokens[0] != "[bos]":
        return "", org_tokens
    elif "[eos]" not in tokens:
        return "", org_tokens

    tokens = tokens[1 : tokens.index("[eos]")]
    if ("[bos]" in tokens) or ("[pad]" in tokens):
        return "", org_tokens
    
    return tokens, org_tokens