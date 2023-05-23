from itertools import product
import numpy as np


PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
UNK_TOKEN = "<unk>"

standard_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
TOKENS = standard_tokens.copy()
# TODO: add tokens (nodes: 현재 상황에서는 그냥 노드 번호인데, 여러 노드를 한 번에 묶게 된다면 (예: [3,4,5] 여기에 토큰 추가해야함))


TOKENS_DICT = {'adj_list': TOKENS}

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

def tokenize(adj_list, string_type):
    
    tokens = ["[bos]"]
    # TODO: tokenize adjacency list
    '''
    여기에서는 adjacency list에 bos, eos 추가하는 역할 + 0,1,2 등을 token으로 매핑 (forward에 들어갈 수 있는 형태)
    (예시 output: [bos], [3,4,5], [0,2], [0,5], [eos])
    '''
    
    tokens.append("[eos]")
    TOKEN2ID = token_to_id(string_type)
    return [TOKEN2ID[token] for token in tokens]


def untokenize(sequence, string_type):
    ID2TOKEN = id_to_token(TOKENS_DICT[string_type])
    tokens = [ID2TOKEN[id_] for id_ in sequence]
    # TODO
    '''
    sequence를 받으면 이것을 다시 adjcency list 형태로 변형 (sampling(generation)할 떄 사용)
    '''
    org_tokens = tokens
    if tokens[0] != "[bos]":
        return "", "".join(org_tokens)
    elif "[eos]" not in tokens:
        return "", "".join(org_tokens)

    tokens = tokens[1 : tokens.index("[eos]")]
    if ("[bos]" in tokens) or ("[pad]" in tokens):
        return "", "".join(org_tokens)
    
    if 'red' in string_type:
        return tokens, org_tokens
    else:
        return "".join(tokens), "".join(org_tokens)