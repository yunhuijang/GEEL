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

dataset_list = ['qm9', 'zinc', 'moses', 'guacamol']
# maximum number of nodes of each dataset (train, test, val)

node_num_dict = {'qm9': 9, 'zinc': 38, 'moses': 31, 'guacamol': 88}
bw_dict = {'qm9': 5, 'zinc': 10, 'moses': 12, 'guacamol': 15}

TOKEN2ATOMFEAT = {
    "[B]": (5, 0, 0),
    "[B-]": (5, -1, 0),
    "[BH2-]": (5, -1, 2),
    "[BH3-]": (5, -1, 3),
    "[BH-]": (5, -1, 1),
    "[CH]": (6, 0, 1),
    "[CH2]": (6, 0, 2),
    "[CH-]": (6, -1, 1),
    "[CH2-]": (6, -1, 2),
    "[C]": (6, 0, 0),
    "[C+]": (6, 1, 0),
    "[CH+]": (6, 1, 1),
    "[CH2+]": (6, 1, 2),
    "[C-]": (6, -1, 0),
    "[N-]": (7, -1, 0),
    "[NH-]": (7, -1, 1),
    "[N]": (7, 0, 0),
    "[NH]": (7, 0, 1),
    "[N+]": (7, 1, 0),
    "[NH+]": (7, 1, 1),
    "[NH2+]": (7, 1, 2),
    "[NH3+]": (7, 1, 3),
    "[O-]": (8, -1, 0),
    "[O]": (8, 0, 0),
    "[O+]": (8, 1, 0),
    "[OH+]": (8, 1, 1),
    "[F]": (9, 0, 0),
    "[F+]": (9, 1, 0),
    "[F-]": (9, -1, 0),
    "[Si]": (14, 0, 0),
    "[Si-]": (14, -1, 0),
    "[SiH-]": (14, -1, 1),
    "[SiH2]": (14, 0, 2),
    "[Si+]": (14, 0, 1),
    "[P]": (15, 0, 0),
    "[PH]": (15, 0, 1),
    "[PH2]": (15, 0, 2),
    "[P+]": (15, 1, 0),
    "[PH+]": (15, 1, 1),
    "[PH2+]": (15, 1, 2),
    "[P-]": (15, -1, 0),
    "[S-]": (16, -1, 0),
    "[SH-]": (16, -1, 1),
    "[S]": (16, 0, 0),
    "[S+]": (16, 1, 0),
    "[SH]": (16, 0, 1),
    "[SH+]": (16, 1, 1),
    "[Cl]": (17, 0, 0),
    "[Cl+]": (17, 1, 0),
    "[Cl-]": (17, -1, 0),
    "[Cl++]": (17, 2, 0),
    "[Cl+++]": (17, 3, 0),
    "[Se]": (34, 0, 0),
    "[Se+]": (34, 1, 0),
    "[Se-]": (34, -1, 0),
    "[SeH]": (34, 0, 1),
    "[SeH2]": (34, 0, 2),
    "[Br]": (35, 0, 0),
    "[Br++]": (35, 2, 0),
    "[Br-]": (35, -1, 0),
    "[I]": (53, 0, 0),
    "[I+]": (53, 1, 0),
    "[IH2]": (53, 0, 2),
    "[I++]": (53, 2, 0),
    "[IH]": (53, 0, 1),
    "[I+++]": (53, 3, 0),
    "[I-]": (53, -1, 0)
}
ATOMFEAT2TOKEN = {val: key for key, val in TOKEN2ATOMFEAT.items()}

TOKENS_DICT_DIFF_NI_MOL = {}

def map_diff(token):
    return (token[0], token[0]-token[1])

def map_diff_ni(token):
    return (token[0], token[1]-token[0])

# TODO: Fix node_tokens_dict for qm9, zinc, and guacamol
NODE_TOKENS_DICT = {'qm9': ['[O]', '[NH+]', '[N]', '[C-]', '[F]', '[NH]', 
                            '[NH3+]', '[N+]', '[C]', '[N-]', '[CH-]', '[NH2+]', '[O-]'], 
                    'zinc': ['[N+]', '[NH3+]', '[SH+]', '[I]', '[O-]', '[NH]', 
                             '[S-]', '[CH2-]', '[NH2+]', '[PH2]', '[CH-]', '[S+]', 
                             '[OH+]', '[S]', '[P+]', '[O]', '[F]', '[C]', '[NH+]', 
                             '[NH-]', '[Cl]', '[N]', '[Br]', '[CH]', '[N-]', '[P]', 
                             '[PH+]', '[O+]', '[PH]'],
                    'moses': ['[CH]', '[CH2]', '[CH-]', '[CH2-]', '[C]', '[N-]', '[NH-]', '[N]', 
                              '[NH]', '[N+]', '[NH+]', '[NH2+]', '[NH3+]', '[O-]', '[O]', '[O+]', 
                              '[OH+]', '[F]', '[P]', '[PH]', '[PH2]', '[P+]', '[PH+]', '[S-]', '[S]', 
                              '[S+]', '[SH]', '[SH+]', '[Cl]', '[Br]', '[I]'], 
                    'guacamol': ['[S-]', '[Si]', '[BH3-]', '[N+]', '[PH]', '[SeH]', '[BH2-]', '[IH2]', 
                                 '[I-]', '[B-]', '[NH]', '[N-]', '[P+]', '[Si+]', '[O-]', '[SH]', '[Se]', 
                                 '[Se+]', '[Cl]', '[C-]', '[I+]', '[SH+]', '[NH+]', '[C]', '[NH-]', '[CH-]', 
                                 '[NH3+]', '[NH2+]', '[OH+]', '[O+]', '[N]', '[C+]', '[S]', '[I]', '[BH-]', 
                                 '[B]', '[P]', '[SH-]', '[S+]', '[O]', '[F]', '[Br]', '[SiH2]']}
bond_tokens = [5,6,7,8]

NODE_TYPE_DICT = {key: idx+9 for idx, key in enumerate(TOKEN2ATOMFEAT.keys())}
BOND_TYPE_DICT = {1: 5, 2: 6, 3: 7, 1.5: 8}

for dataset in ['qm9', 'zinc', 'moses', 'guacamol']:
    
    bw = bw_dict[dataset]
    node_num = node_num_dict[dataset]

    
    tokens_list_diff_node_edge_ni = standard_tokens.copy()
    tokens_list_diff_node_edge_ni.extend([(num, b) for b in np.arange(0,bw+1) for num in np.arange(0,2)])
    tokens_list_diff_node_edge_ni.remove((0,0))
    tokens_list_diff_node_edge_ni.extend(NODE_TOKENS_DICT[dataset])
    tokens_list_diff_node_edge_ni.extend(bond_tokens)
    TOKENS_DICT_DIFF_NI_MOL[dataset] = tokens_list_diff_node_edge_ni
    

def token_list_to_dict(tokens):
    return {token: i for i, token in enumerate(tokens)}

TOKENS_KEY_DICT_DIFF_NI_MOL = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_DIFF_NI_MOL.items()}

def token_to_id_mol(data_name, string_type):
    return TOKENS_KEY_DICT_DIFF_NI_MOL[data_name]

def id_to_token_mol(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def tokenize_mol(adj, adj_list, node_attr, edge_attr, data_name, string_type):
    TOKEN2ID = token_to_id_mol(data_name, string_type)
    tokens = ["[bos]"]    
    if string_type == 'adj_list_diff_ni':
        reverse_adj_list = [(tar, src) for src, tar in adj_list]
        src_node_set = set([src for src, _ in reverse_adj_list])
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
    else:
        raise ValueError('String type must be adj_list_diff_ni for molecules')

    tokens.append("[eos]")
    # return tokens
    return [TOKEN2ID[token] for token in tokens]


def untokenize_mol(sequence, data_name, string_type, is_token, vocab_size=200):
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

