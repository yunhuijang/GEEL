import torch
import numpy as np

from data.data_utils import get_max_len


PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
UNK_TOKEN = "<unk>"

standard_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

dataset_list = ['GDSS_ego', 'GDSS_com', 'GDSS_enz', 'GDSS_grid', 'planar', 'sbm', 'proteins']
# dataset_list = ['GDSS_ego']
# maximum number of nodes of each dataset (train, test, val)
node_num_list = [17, 20, 125, 361, 64, 187, 500]
bw_list = [15, 8, 19, 19, 26, 111, 499]

TOKENS_DICT = {}
TOKENS_DICT_DIFF = {}
TOKENS_DICT_FEATURED = {}
TOKENS_DICT_FLATTEN = {}
TOKENS_DICT_SEQ = {}
# map adj_list tokens
for dataset, node_num in zip(dataset_list, node_num_list):
    tokens = standard_tokens.copy()
    tokens.extend([(i,j) for i in range(node_num) for j in range(i+1, node_num+1)])
    TOKENS_DICT[dataset] = tokens

def map_diff(token):
    return (token[0], token[1]-token[0])

for dataset, tokens in TOKENS_DICT.items():
    # map adj_list_diff tokens
    tokens_diff = standard_tokens.copy()
    tokens_diff.extend([map_diff(token) for token in tokens if type(token) is tuple])
    TOKENS_DICT_DIFF[dataset] = tokens_diff
    # map adj_flatten / adj_flatten_sym tokens
    tokens_flat = standard_tokens.copy()
    tokens_flat.extend([0, 1])
    TOKENS_DICT_FLATTEN[dataset] = tokens_flat

# map sequential representation tokens
for dataset, bw in zip(dataset_list, bw_list):
    tokens_seq = standard_tokens.copy()
    # 0: node token
    tokens_seq.append(0)
    # 1-n: edge relative position token
    tokens_seq.extend(np.arange(1,2*bw))
    TOKENS_DICT_SEQ[dataset] = tokens_seq
    
# for dataset, num_nodes, num_node_types, num_edge_types in zip(['qm9', 'zinc'], ,):
#     tokens_featured = standard_tokens.copy()
#     tokens_featured 

def token_list_to_dict(tokens):
    return {token: i for i, token in enumerate(tokens)}

TOKENS_KEY_DICT = {key: token_list_to_dict(value) for key, value in TOKENS_DICT.items()}
TOKENS_KEY_DICT_DIFF = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_DIFF.items()}
TOKENS_KEY_DICT_FLATTEN = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_FLATTEN.items()}
TOKENS_KEY_DICT_SEQ = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_SEQ.items()}

def token_to_id(data_name, string_type):
    if string_type == 'adj_list':
        return TOKENS_KEY_DICT[data_name]
    elif string_type == 'adj_list_diff':
        return TOKENS_KEY_DICT_DIFF[data_name]
    elif string_type in ['adj_flatten', 'adj_flatten_sym']:
        return TOKENS_KEY_DICT_FLATTEN[data_name]
    elif string_type in ['adj_seq']:
        return TOKENS_KEY_DICT_SEQ[data_name]

def id_to_token(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def tokenize(adj, adj_list, data_name, string_type):

    TOKEN2ID = token_to_id(data_name, string_type)
    tokens = ["[bos]"]
    if string_type == 'adj_list':
        tokens.extend(adj_list)
    elif string_type == 'adj_list_diff':
        tokens.extend([map_diff(edge) for edge in adj_list])
    elif string_type == 'adj_flatten':
        tokens.extend(torch.flatten(torch.tensor(adj.todense())).tolist())
    elif string_type == 'adj_flatten_sym':
        np_adj = adj.toarray()
        lower_diagonal = np_adj[np.tril_indices(len(np_adj))]
        tokens.extend(lower_diagonal.tolist())
    elif string_type == 'adj_seq':
        tokens.append(0)
        prev_src_node = 1
        for src_node, tar_node in adj_list:
            if prev_src_node != src_node:
                tokens.append(0)
            diff = src_node - tar_node
            tokens.append(diff)
            prev_src_node = src_node
    elif string_type == 'adj_seq_rel':
        tokens.append(0)
        prev_src_node = 1
        adj_list = sorted(adj_list, key = lambda x: (x[0], -x[1]))
        for src_node, tar_node in adj_list:
            if prev_src_node != src_node:
                tokens.append(0)
                cur_tar_node = tar_node
            diff = cur_tar_node - tar_node
            tokens.append(diff)
            prev_src_node = src_node
            cur_tar_node = tar_node
        
    tokens.append("[eos]")

    return [TOKEN2ID[token] for token in tokens]


def untokenize(sequence, data_name, string_type):
    if string_type == 'adj_list':
        tokens = TOKENS_DICT[data_name]
    elif string_type == 'adj_list_diff':
        tokens = TOKENS_DICT_DIFF[data_name]
    elif string_type in ['adj_flatten', 'adj_flatten_sym']:
        tokens = TOKENS_DICT_FLATTEN[data_name]
    elif string_type == 'adj_seq':
        tokens = TOKENS_DICT_SEQ[data_name]
        
    ID2TOKEN = id_to_token(tokens)
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