from itertools import product

PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
UNK_TOKEN = "<unk>"

TOKENS_DFS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    "(",
    ")", 
    '1',
    '0'
]
TOKENS_BFS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    '1',
    '0'
]

TOKENS_GROUP = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN
]

group_num_tokens = list(product([0,1], repeat=4))
TOKENS_GROUP.extend([''.join(str(token)).replace(', ', '')[1:-1] for token in group_num_tokens])

TOKENS_BFS_DEG = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    '0',
    '1',
    '2',
    '3',
    '4'
]


TOKENS_DICT = {'dfs': TOKENS_DFS, 'bfs': TOKENS_BFS, 'group': TOKENS_GROUP, 'bfs-deg': TOKENS_BFS_DEG}

def token_to_id(tokens):
    return {token: tokens.index(token) for token in tokens}

def id_to_token(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def tokenize(string, string_type):
    tokens = ["[bos]"]
    if string_type == 'group':
        string_cut = [string[i:i+4] for i in range(0, len(string), 4)]
        tokens.extend([*string_cut])
    else:
        tokens.extend([*string])
    tokens.append("[eos]")
    TOKEN2ID = token_to_id(TOKENS_DICT[string_type])
    return [TOKEN2ID[token] for token in tokens]

def untokenize(sequence, string_type):
    ID2TOKEN = id_to_token(TOKENS_DICT[string_type])
    tokens = [ID2TOKEN[id_] for id_ in sequence]
    if string_type == 'bfs-deg':
        mapping_dict = {'2': '1', '3': '1', '4': '1'}
        tokens = [mapping_dict.get(token, token) for token in tokens]
    org_tokens = tokens
    if tokens[0] != "[bos]":
        return "", "".join(org_tokens)
    elif "[eos]" not in tokens:
        return "", "".join(org_tokens)

    tokens = tokens[1 : tokens.index("[eos]")]
    if ("[bos]" in tokens) or ("[pad]" in tokens):
        return "", "".join(org_tokens)
    
    return "".join(tokens), "".join(org_tokens)
    
    