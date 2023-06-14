from itertools import product
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

TOKENS_DICT = {}
TOKENS_DICT_DIFF = {}
# map adj_list tokens
for dataset, node_num in zip(dataset_list, node_num_list):
    tokens = standard_tokens.copy()
    tokens.extend([(i,j) for i in range(node_num) for j in range(i+1, node_num+1)])
    TOKENS_DICT[dataset] = tokens

def map_diff(token):
    return (token[0], token[1]-token[0])

for dataset, tokens in TOKENS_DICT.items():
    tokens_diff = standard_tokens.copy()
    tokens_diff.extend([map_diff(token) for token in tokens if type(token) is tuple])
    TOKENS_DICT_DIFF[dataset] = tokens_diff

def token_list_to_dict(tokens):
    return {token: i for i, token in enumerate(tokens)}

TOKENS_KEY_DICT = {key: token_list_to_dict(value) for key, value in TOKENS_DICT.items()}

TOKENS_KEY_DICT_DIFF = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_DIFF.items()}

def token_to_id(data_name, string_type):
    if string_type == 'adj_list':
        return TOKENS_KEY_DICT[data_name]
    elif string_type == 'adj_list_diff':
        return TOKENS_KEY_DICT_DIFF[data_name]

def id_to_token(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def tokenize(adj_list, data_name, string_type):

    TOKEN2ID = token_to_id(data_name, string_type)
    tokens = ["[bos]"]
    if string_type == 'adj_list':
        tokens.extend(adj_list)
    elif string_type == 'adj_list_diff':
        tokens.extend([map_diff(edge) for edge in adj_list])
    tokens.append("[eos]")

    return [TOKEN2ID[token] for token in tokens]


def untokenize(sequence, data_name, string_type):
    if string_type == 'adj_list':
        tokens = TOKENS_DICT[data_name]
    elif string_type == 'adj_list_diff':
        tokens = TOKENS_DICT_DIFF[data_name]
        
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