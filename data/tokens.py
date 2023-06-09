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
for dataset, node_num in zip(dataset_list, node_num_list):
    # node_num = get_max_len(dataset)[1]
    tokens = standard_tokens.copy()
    tokens.extend([(i,j) for i in range(node_num) for j in range(i+1, node_num+1)])
    TOKENS_DICT[dataset] = tokens

def token_list_to_dict(tokens):
    return {token: i for i, token in enumerate(tokens)}

TOKENS_KEY_DICT = {key: token_list_to_dict(value) for key, value in TOKENS_DICT.items()}

def token_to_id(data_name):
    return TOKENS_KEY_DICT[data_name]

def id_to_token(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def tokenize(adj_list, data_name):

    tokens = ["[bos]"]
    tokens.extend(adj_list)
    tokens.append("[eos]")
    TOKEN2ID = token_to_id(data_name)
    # if data_name == "GDSS_com":
    #     TOKEN2ID = token_to_id('adj_list_com')
    # elif data_name == "GDSS_ego":
    #     TOKEN2ID = token_to_id('adj_list_ego')
    # elif data_name == "GDSS_enz":
    #     TOKEN2ID = token_to_id('adj_list_enz')
    # elif data_name == "GDSS_grid":
    #     TOKEN2ID = token_to_id('adj_list_grid')
    
    return [TOKEN2ID[token] for token in tokens]


def untokenize(sequence, data_name):

    ID2TOKEN = id_to_token(TOKENS_DICT[data_name])
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