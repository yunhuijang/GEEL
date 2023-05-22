from itertools import product
import numpy as np


PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
UNK_TOKEN = "<unk>"

standard_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
TOKENS_BFS = standard_tokens.copy()
TOKENS_BFS.extend([str(x) for x in np.arange(0, 2)])

TOKENS_BFS_DEG = standard_tokens.copy()
TOKENS_BFS_DEG.extend([str(x) for x in np.arange(0, 5)])

TOKENS_DFS = standard_tokens.copy()
TOKENS_DFS.extend(["(", ")", '0', '1'])

TOKENS_GROUP = standard_tokens.copy()
group_num_tokens = list(product([0,1], repeat=4))
TOKENS_GROUP.extend([''.join(str(token)).replace(', ', '')[1:-1] for token in group_num_tokens if token!=(0,0,0,0)])

TOKENS_BFS_DEG_GROUP = standard_tokens.copy()
group_num_tokens = list(product([0,1,2,3,4], repeat=4))
TOKENS_BFS_DEG_GROUP.extend([''.join(str(token)).replace(', ', '')[1:-1] for token in group_num_tokens])

TOKENS_GROUP_RED = TOKENS_GROUP.copy()
group_num_tokens = list(product([0,1], repeat=3))
TOKENS_GROUP_RED.extend([''.join(str(token)).replace(', ', '')[1:-1] for token in group_num_tokens if token!=(0,0,0)])

TOKENS_GROUP_RED_DICT = {}
TOKENS_GROUP_RED_DICT[2] = TOKENS_GROUP_RED
TOKENS_GROUP_THREE = standard_tokens.copy()
group_num_tokens = list(product([0,1], repeat=9))
TOKENS_GROUP_THREE.extend([''.join(str(token)).replace(', ', '')[1:-1] for token in group_num_tokens if token!=tuple(np.zeros(9, dtype=int))])
for length in range(1,int(3*(3-1)/2)+1):
    group_num_tokens = list(product([0,1], repeat=9-length))
    TOKENS_GROUP_THREE.extend([''.join(str(token)).replace(', ', '')[1:-1] for token in group_num_tokens if token!=tuple(np.zeros(9-length, dtype=int))])
TOKENS_GROUP_RED_DICT[3] = TOKENS_GROUP_THREE


def grouper_mol(string, k):
    non_group_tokens = []
    k_square = k**2
    string_iter = iter(string)
    peek = None
    while True:
        char = peek if peek else next(string_iter, "")
        peek = None
        if not char:
            break
        if char in ['C', 'B']:
            peek = next(string_iter, "")
            if char + peek in ['Cl', 'Br']:
                token = char + peek
                peek = None
            else:
                token = char
        else:
            token = char
        non_group_tokens.append(token)
    string_cut = [non_group_tokens[i:i+k_square] for i in range(0,len(non_group_tokens),k_square)]
    cut_list = [*string_cut]
    return cut_list


TOKENS_MOL = TOKENS_GROUP.copy()
# 5: single / 6: double / 7:triple / 8: aromatic
bond_tokens = [0,5,6,7,8]
# mol_bond_tokens: only edge type + 0s (without diagonal)
mol_bond_tokens = list(product(bond_tokens, repeat=4))
TOKENS_MOL.extend([''.join(str(token)).replace(', ', '')[1:-1] for token in mol_bond_tokens])

node_tokens_dict = {'qm9': ['F', 'O', 'N', 'C'], 'zinc': ['F', 'O', 'N', 'C', 'P', 'I', 'Cl', 'Br', 'S']}
additional_tokens_dict = dict()

for data in ['qm9', 'zinc']:
    node_tokens = node_tokens_dict[data]
    additional_tokens = list(product(node_tokens, bond_tokens, bond_tokens, node_tokens))
    additional_tokens.extend(list(product(node_tokens, [0], [0], [0])))
    additional_tokens.extend(list(product(bond_tokens, node_tokens, [0], [0])))
    additional_tokens.extend(list(product(bond_tokens, [0], node_tokens, [0])))
    additional_tokens.extend(list(product(bond_tokens, [0], node_tokens, bond_tokens)))
    # additional_tokens.extend(list(product(bond_tokens, node_tokens, [0], bond_tokens)))
    additional_tokens_dict[data] = additional_tokens

TOKENS_QM9 = TOKENS_MOL.copy()
TOKENS_QM9.extend([''.join(str(token)).replace(', ', '').replace('\'', '')[1:-1] for token in additional_tokens_dict['qm9']])
TOKENS_ZINC = TOKENS_MOL.copy()
TOKENS_ZINC.extend([''.join(str(token)).replace(', ', '').replace('\'', '')[1:-1] for token in additional_tokens_dict['zinc']])

TOKENS_MOL_RED = TOKENS_GROUP_RED.copy()
TOKENS_MOL_RED.extend([''.join(str(token)).replace(', ', '')[1:-1] for token in mol_bond_tokens])
mol_bond_tokens_red = list(product(bond_tokens, repeat=3))
TOKENS_MOL_RED.extend([''.join(str(token)).replace(', ', '')[1:-1] for token in mol_bond_tokens_red])

TOKENS_QM9_RED = TOKENS_MOL_RED.copy()
qm9_additional_tokens = [''.join(str(token)).replace(', ', '').replace('\'', '')[1:-1] for token in additional_tokens_dict['qm9']]
TOKENS_QM9_RED.extend(qm9_additional_tokens)
TOKENS_QM9_RED.extend([token[:1]+token[2:] for token in qm9_additional_tokens])

TOKENS_ZINC_RED = TOKENS_MOL_RED.copy()
zinc_additional_tokens = [''.join(str(token)).replace(', ', '').replace('\'', '')[1:-1] for token in additional_tokens_dict['zinc']]
TOKENS_ZINC_RED.extend(zinc_additional_tokens)
group_ad_tokens = [grouper_mol(token, 2)[0] for token in zinc_additional_tokens]
for group in group_ad_tokens:
    del group[1]

TOKENS_ZINC_RED.extend(list(set([''.join(token) for token in group_ad_tokens])))

TOKENS_DICT = {'dfs': TOKENS_DFS, 'bfs': TOKENS_BFS, 'group': TOKENS_GROUP, 
               'bfs-deg': TOKENS_BFS_DEG, 'bfs-deg-group': TOKENS_BFS_DEG_GROUP,
               'qm9': TOKENS_QM9, 'zinc': TOKENS_ZINC, 'group-red-2': TOKENS_GROUP_RED, 
               'group-red-3': TOKENS_GROUP_THREE, 'qm9-red': TOKENS_QM9_RED, 'zinc-red': TOKENS_ZINC_RED}


def token_list_to_dict(tokens):
    return {token: i for i, token in enumerate(tokens)}

TOKENS_KEY_DICT = {key: token_list_to_dict(value) for key, value in TOKENS_DICT.items()}

def token_to_id(string_type, k=2):
    return TOKENS_KEY_DICT[string_type]

def id_to_token(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def tokenize(string, string_type, k):
    tokens = ["[bos]"]
    if string_type in ['group', 'bfs-deg-group', 'qm9']:
        string_cut = [string[i:i+4] for i in range(0, len(string), 4)]
        tokens.extend([*string_cut])
    elif string_type in ['zinc']:
        cut_list = grouper_mol(string)
        tokens.extend([''.join(tok) for tok in cut_list])
    elif 'red' in string_type:
        tokens.extend(string)
    else:
        tokens.extend([*string])
    tokens.append("[eos]")
    TOKEN2ID = token_to_id(string_type, k)
    return [TOKEN2ID[token] for token in tokens]

def map_one(token):
    mapping_dict = {'2': '1', '3': '1', '4': '1'}
    return ''.join([mapping_dict.get(x, x) for x in token])

def untokenize(sequence, string_type, k=2):
    ID2TOKEN = id_to_token(TOKENS_DICT[string_type])
    tokens = [ID2TOKEN[id_] for id_ in sequence]
    if string_type in ['bfs-deg', 'bfs-deg-group']:
        tokens = [map_one(token) for token in tokens]
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