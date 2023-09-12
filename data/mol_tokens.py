import torch
import numpy as np
from itertools import product

from data.data_utils import flatten_forward
from data.orderings import bw_from_adj


PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
UNK_TOKEN = "<unk>"

standard_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

dataset_list = ['qm9', 'zinc', 'moses']
# maximum number of nodes of each dataset (train, test, val)
node_num_dict = {'qm9': 9, 'zinc': 38, 'moses': 26}
bw_dict = {'qm9': 5, 'zinc': 10, 'moses': 6}

TOKENS_DICT_MOL = {}
TOKENS_DICT_FLATTEN_MOL = {}
TOKENS_DICT_SEQ_MOL = {}
TOKENS_DICT_SEQ_MERGE_MOL = {}
TOKENS_DICT_DIFF_MOL  = {}
TOKENS_DICT_DIFF_NI_MOL = {}

def map_diff(token):
    return (token[0], token[0]-token[1])

def map_diff_ni(token):
    return (token[0], token[1]-token[0])


NODE_TOKENS_DICT = {'qm9': ['F', 'O', 'N', 'C'], 'zinc': ['F', 'O', 'N', 'C', 'P', 'I', 'Cl', 'Br', 'S'],
                    'moses': ['F', 'O', 'N', 'C', 'Cl', 'Br', 'S']}
bond_tokens = [5,6,7,8]

NODE_TYPE_DICT = {'F': 9, 'O': 10, 'N': 11, 'C': 12, 'P': 13, 'I': 14, 'Cl': 15, 'Br': 16, 'S': 17}
TYPE_NODE_DICT = {str(key): value for value, key in NODE_TYPE_DICT.items()}
BOND_TYPE_DICT = {1: 5, 2: 6, 3: 7, 1.5: 8}
TYPE_BOND_DICT = {key: value for value, key in NODE_TYPE_DICT.items()}

for dataset in ['qm9', 'zinc', 'moses']:
    
    bw = bw_dict[dataset]
    node_num = node_num_dict[dataset]
    
    # tokens = standard_tokens.copy()
    # tokens.extend(NODE_TYPE_DICT[node_type] for node_type in NODE_TOKENS_DICT[dataset])
    # tokens.extend([(src_bond, tar_bond) for src_bond, tar_bond in product(bond_tokens, bond_tokens)])
    # TOKENS_DICT_MOL[dataset] = tokens
    
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
    
    tokens_seq_merge = standard_tokens.copy()
    node_types = [NODE_TYPE_DICT[node_type] for node_type in NODE_TOKENS_DICT[dataset]]
    edge_types = BOND_TYPE_DICT.values()
    seq_tokens = np.arange(1, bw_dict[dataset]+1)
    node_tokens = [(0, node_type) for node_type in node_types]
    edge_tokens = [(seq_token, edge_type) for seq_token, edge_type in product(seq_tokens, edge_types)]
    tokens_seq_merge.extend(node_tokens)
    tokens_seq_merge.extend(edge_tokens)
    TOKENS_DICT_SEQ_MERGE_MOL[dataset] = tokens_seq_merge
    
    tokens_list_node_edge = standard_tokens.copy()
    tokens_list_node_edge.extend([(num, num-b) for b in np.arange(1,bw+1) for num in np.arange(1,node_num) if (num-b >= 0)])
    tokens_list_node_edge.extend(NODE_TYPE_DICT[node_type] for node_type in NODE_TOKENS_DICT[dataset])
    tokens_list_node_edge.extend(bond_tokens)
    TOKENS_DICT_MOL[dataset] = tokens_list_node_edge
    
    tokens_list_diff_node_edge = standard_tokens.copy()
    tokens_list_diff_node_edge.extend([(num, b) for b in np.arange(1,bw+1) for num in np.arange(1,node_num) if (num-b >= 0)])
    tokens_list_diff_node_edge.extend(NODE_TYPE_DICT[node_type] for node_type in NODE_TOKENS_DICT[dataset])
    tokens_list_diff_node_edge.extend(bond_tokens)
    TOKENS_DICT_DIFF_MOL[dataset] = tokens_list_diff_node_edge
    
    tokens_list_diff_node_edge_ni = standard_tokens.copy()
    tokens_list_diff_node_edge_ni.extend([(num, b) for b in np.arange(0,bw+1) for num in np.arange(0,2)])
    tokens_list_diff_node_edge_ni.remove((0,0))
    tokens_list_diff_node_edge_ni.extend(NODE_TYPE_DICT[node_type] for node_type in NODE_TOKENS_DICT[dataset])
    tokens_list_diff_node_edge_ni.extend(bond_tokens)
    TOKENS_DICT_DIFF_NI_MOL[dataset] = tokens_list_diff_node_edge_ni
    

def token_list_to_dict(tokens):
    return {token: i for i, token in enumerate(tokens)}

TOKENS_KEY_DICT_MOL = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_MOL.items()}
TOKENS_KEY_DICT_DIFF_MOL = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_DIFF_MOL.items()}
TOKENS_KEY_DICT_FLATTEN_MOL = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_FLATTEN_MOL.items()}
TOKENS_KEY_DICT_SEQ_MOL = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_SEQ_MOL.items()}
TOKENS_KEY_DICT_SEQ_MERGE_MOL = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_SEQ_MERGE_MOL.items()}
TOKENS_KEY_DICT_DIFF_NI_MOL = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_DIFF_NI_MOL.items()}

def token_to_id_mol(data_name, string_type):
    if string_type in ['adj_list']:
        return TOKENS_KEY_DICT_MOL[data_name]
    elif string_type == 'adj_list_diff':
        return TOKENS_KEY_DICT_DIFF_MOL[data_name]
    elif string_type in ['adj_flatten', 'adj_flatten_sym', 'bwr']:
        return TOKENS_KEY_DICT_FLATTEN_MOL[data_name]
    elif string_type in ['adj_seq', 'adj_seq_rel']:
        return TOKENS_KEY_DICT_SEQ_MOL[data_name]
    elif string_type in ['adj_seq_merge', 'adj_seq_rel_merge']:
        return TOKENS_KEY_DICT_SEQ_MERGE_MOL[data_name]
    elif string_type == 'adj_list_diff_ni':
        return TOKENS_KEY_DICT_DIFF_NI_MOL[data_name]

def id_to_token_mol(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def tokenize_mol(adj, adj_list, node_attr, edge_attr, data_name, string_type):
    TOKEN2ID = token_to_id_mol(data_name, string_type)
    tokens = ["[bos]"]
    if string_type in ['adj_list', 'adj_list_diff']:
        edge_attr_reverse = {(key[1], key[0]): value for key, value in edge_attr.items()}
        tokens.append(node_attr[0])
        prev_src_node = 0
        for edge in adj_list:
            cur_src_node = edge[0]
            if cur_src_node != prev_src_node:
                tokens.append(node_attr[cur_src_node])
            if string_type == 'adj_list':
                tokens.append(edge)
            else:
                edge_diff = map_diff(edge)
                tokens.append(edge_diff)
            tokens.append(edge_attr_reverse[edge])
            prev_src_node = cur_src_node
    elif string_type == 'adj_list_diff_ni':
        edge_attr_reverse = {(key[1], key[0]): value for key, value in edge_attr.items()}
        reverse_adj_list = [(tar, src) for src, tar in adj_list]
        src_node_set = set([src for src, tar in reverse_adj_list])
        for node in node_attr.keys():
            if node not in src_node_set:
                reverse_adj_list.append((node, node))
        reverse_adj_list = sorted(reverse_adj_list, key=lambda x: x[0])
        # tokens.append(node_attr[0])
        prev_src_node = -1
        for edge in reverse_adj_list:
            src_node = edge[0]
            tar_node = edge[1]
            cur_src_node = src_node
            if cur_src_node != prev_src_node:
                tokens.append(node_attr[cur_src_node])
                final_src_node = 1
            else:
                final_src_node = 0
            if src_node == tar_node:
                tokens.append((final_src_node, 0))
                # virtual edge type
                tokens.append(5)
            else:
                edge_diff = map_diff_ni(edge)
                tokens.append((final_src_node, edge_diff[1]))
                tokens.append(edge_attr[edge])
            prev_src_node = cur_src_node
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
    elif string_type == 'adj_seq_merge':
        tokens.append((0, node_attr[0]))
        prev_src_node = 0
        for src_node, tar_node in adj_list:
            if prev_src_node != src_node:
                tokens.append((0, node_attr[src_node]))
            diff = src_node - tar_node
            tokens.append((diff, edge_attr[(tar_node, src_node)]))
            prev_src_node = src_node
    elif string_type == 'adj_seq_rel_merge':
        prev_src_node = 0
        adj_list = sorted(adj_list, key = lambda x: (x[0], -x[1]))
        cur_tar_node = adj_list[0][1]
        tokens.append((0, node_attr[0]))
        for src_node, tar_node in adj_list:
            if prev_src_node != src_node:
                tokens.append((0, node_attr[src_node]))
                diff = src_node - tar_node
            else:
                diff = cur_tar_node - tar_node
            if diff != 0:
                tokens.append((diff, edge_attr[(tar_node, src_node)]))
            prev_src_node = src_node
            cur_tar_node = tar_node
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
    elif string_type in ['adj_seq_rel_merge', 'adj_seq_merge']:
        tokens = TOKENS_DICT_SEQ_MERGE_MOL[data_name]
    elif string_type == 'adj_list_diff_ni':
        tokens = TOKENS_DICT_DIFF_NI_MOL[data_name]
        
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