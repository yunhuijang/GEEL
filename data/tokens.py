import numpy as np
import json

from data.mol_tokens import TOKENS_DICT_DIFF_NI_MOL


PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
UNK_TOKEN = "<unk>"

standard_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

# TODO: fix moses / guacamol node_num and bw

dataset_list = ['GDSS_ego', 'GDSS_com', 'GDSS_enz', 'GDSS_grid', 'planar', 'sbm', 
                'proteins', 'lobster', 'point', 'ego', 'qm9', 'zinc', 'moses', 'guacamol',
                'grid-500', 'grid-1000', 'grid-2000', 'grid-5000', 'grid-10000', 'grid-20000', 'grid-50000', 'grid-100000']
# maximum number of nodes of each dataset (train, test, val)
node_num_list = [17, 20, 125, 361, 64, 187,
                 500, 98, 5037, 399, 9, 38, 31, 88,
                 676, 1296, 2401, 5620, 10816, 21025, 51984]
bw_list = [15, 8, 19, 19, 26, 111, 
           125, 49, 167, 241, 5, 10, 12, 12,
           26, 36, 49, 75, 104, 145, 228]

TOKENS_DICT = {}
TOKENS_DICT_DIFF_NI = {}
TOKENS_DICT_DIFF_NI_REL = {}

def map_diff(token):
    return (token[0], token[0]-token[1])

def map_diff_ni(token):
    return (token[0], token[1]-token[0])

def map_diff_ni_tokens(dataset, order):
    
    with open("resource/graph_info.json", "r") as json_file:
        json_data = json.load(json_file)
    
    graph_info = json_data[dataset]
    bw = graph_info['bw'][order]
    
    diff_ni_tokens = standard_tokens.copy()
    diff_ni_tokens.extend([(num, b) for b in np.arange(0,bw+1) for num in np.arange(0,2)])
    
    return diff_ni_tokens

def map_diff_ni_rel_tokens(dataset, order):
    with open("resource/graph_info.json", "r") as json_file:
        json_data = json.load(json_file)
    
    graph_info = json_data[dataset]
    bw = graph_info['bw'][order]
    
    diff_ni_rel_tokens = standard_tokens.copy()
    diff_ni_rel_tokens.extend([(num, b) for b in np.arange(1,bw+1) for num in np.arange(0,bw+1)])
    
    return diff_ni_rel_tokens

# map sequential representation tokens
for dataset, bw, node_num in zip(dataset_list, bw_list, node_num_list):
    # map adj_list tokens
    tokens = standard_tokens.copy()
    tokens.extend([(num, num-b) for b in np.arange(1,bw+1) for num in np.arange(1,node_num) if (num-b >= 0)])
    TOKENS_DICT[dataset] = tokens
    
    # map token_list_diff_ni tokens (src node: NI (0,1), tar node: difference)
    tokens_diff_ni = standard_tokens.copy()
    tokens_diff_ni.extend([(num, b) for b in np.arange(0,bw+1) for num in np.arange(0,2)])
    TOKENS_DICT_DIFF_NI[dataset] = tokens_diff_ni
    
    # map token_list_diff_ni_rel tokens (src node: NI (0,1, ..., BW), tar node: difference)
    tokens_diff_ni_rel = standard_tokens.copy()
    tokens_diff_ni_rel.extend([(num, b) for b in np.arange(1,bw+1) for num in np.arange(0,bw+1)])
    TOKENS_DICT_DIFF_NI_REL[dataset] = tokens_diff_ni_rel

      
def token_list_to_dict(tokens):
    return {token: i for i, token in enumerate(tokens)}

TOKENS_KEY_DICT = {key: token_list_to_dict(value) for key, value in TOKENS_DICT.items()}
TOKENS_KEY_DICT_DIFF_NI = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_DIFF_NI.items()}
TOKENS_KEY_DICT_DIFF_NI_REL = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_DIFF_NI_REL.items()}

def token_to_id(data_name, string_type, is_token=False, vocab_size=200, order='C-M'):
    if string_type in ['adj_list_diff_ni','adj_list_diff_ni_rel']:
        tokens = map_tokens(data_name, string_type, 0, order)
        return token_list_to_dict(tokens)
    else:
        assert False, "No token type"

def id_to_token(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def tokenize(adj, adj_list, data_name, string_type, is_token=False, vocab_size=200, order='C-M'):
    TOKEN2ID = token_to_id(data_name, string_type, is_token, vocab_size, order)
    tokens = ["[bos]"]

    if string_type not in ['adj_list_diff_ni', 'adj_list_diff_ni_rel']:
        raise ValueError('String type must be adj_list_diff_ni or adj_list_diff_ni_rel')
    else:
        reverse_adj_list = [(tar, src) for src, tar in adj_list]
        reverse_adj_list = sorted(reverse_adj_list, key=lambda x: x[0])
        adj_diff_list = [map_diff_ni(edge) for edge in reverse_adj_list]
        src_node_set = set([src for src, tar in adj_diff_list])
        if string_type == 'adj_list_diff_ni':
            # add self-loop
            for node in range(max(src_node_set)+1):
                if node not in src_node_set:
                    adj_diff_list.append((node, 0))
        adj_diff_list = sorted(adj_diff_list, key=lambda x: x[0])
        prev_src_node = -1
        for src_node, tar_node in adj_diff_list:
            if prev_src_node != src_node:
                if string_type == 'adj_list_diff_ni':
                    final_src_node = 1
                else:
                    final_src_node = src_node - prev_src_node
            else:
                final_src_node = 0
            tokens.append((final_src_node, tar_node))
            prev_src_node = src_node
    
    tokens.append("[eos]")

    return [TOKEN2ID[token] for token in tokens]


def untokenize(sequence, data_name, string_type, is_token, order, vocab_size=200):
    tokens = map_tokens(data_name, string_type, vocab_size, order, is_token)
        
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

def map_tokens(data_name, string_type, vocab_size, order, is_token=False):
    if string_type == 'adj_list_diff_ni':
        if data_name in ['qm9', 'zinc', 'moses', 'guacamol']:
            tokens = TOKENS_DICT_DIFF_NI_MOL[data_name]
        else:
            tokens = map_diff_ni_tokens(dataset=data_name, order=order)
    elif string_type == 'adj_list_diff_ni_rel':
        tokens = map_diff_ni_rel_tokens(dataset=data_name, order=order)

    else:
        assert False, "No token type"
    return tokens