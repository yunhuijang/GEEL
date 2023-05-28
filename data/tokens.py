from itertools import product
import numpy as np


PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
UNK_TOKEN = "<unk>"

standard_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

#com
TOKENS_COM = standard_tokens.copy()
for i in range(20-1):
    for j in range(i+1, 20):
        TOKENS_COM.append((i,j))

#ego
TOKENS_EGO = standard_tokens.copy()
for i in range(17-1):
    for j in range(i+1, 17):
        TOKENS_EGO.append((i,j))

#enz
TOKENS_ENZ = standard_tokens.copy()
for i in range(125-1):
    for j in range(i+1, 125):
        TOKENS_ENZ.append((i,j))
        
#grid
TOKENS_GRID = standard_tokens.copy()
for i in range(361-1):
    for j in range(i+1, 361):
        TOKENS_GRID.append((i,j))

TOKENS_DICT = {'adj_list_com': TOKENS_COM, 'adj_list_ego': TOKENS_EGO, 'adj_list_enz': TOKENS_ENZ, 'adj_list_grid': TOKENS_GRID}

def token_list_to_dict(tokens):
    return {token: i for i, token in enumerate(tokens)}

TOKENS_KEY_DICT = {key: token_list_to_dict(value) for key, value in TOKENS_DICT.items()}

def token_to_id(string_type):
    '''
    string_type은 adj_list로 고정
    '''
    return TOKENS_KEY_DICT[string_type]

def id_to_token(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def tokenize(adj_list, data_name):

    tokens = ["[bos]"]
    tokens.extend(adj_list)
    tokens.append("[eos]")

    if data_name == "GDSS_com":
        TOKEN2ID = token_to_id('adj_list_com')
    elif data_name == "GDSS_ego":
        TOKEN2ID = token_to_id('adj_list_ego')
    elif data_name == "GDSS_enz":
        TOKEN2ID = token_to_id('adj_list_enz')
    elif data_name == "GDSS_grid":
        TOKEN2ID = token_to_id('adj_list_grid')
    
    return [TOKEN2ID[token] for token in tokens]


def untokenize(sequence, string_type):

    ID2TOKEN = id_to_token(TOKENS_DICT[string_type])
    tokens = [ID2TOKEN[id_] for id_ in sequence]

    org_tokens = tokens
    if tokens[0] != "[bos]":
        return "", org_tokens
    elif "[eos]" not in tokens:
        return "", org_tokens

    tokens = tokens[1 : tokens.index("[eos]")]
    if ("[bos]" in tokens) or ("[pad]" in tokens):
        return "", org_tokens
    
    if 'red' in string_type:
        return tokens, org_tokens
    else:
        return tokens, org_tokens