import torch
import numpy as np
import json
import sentencepiece as spm
from collections import defaultdict

from data.data_utils import flatten_forward, map_string_adj_seq, map_string_adj_seq_rel, map_string_flat_sym, map_string_adj_seq_blank, map_string_adj_seq_rel_blank, adj_list_diff_ni_to_adj_list
from data.orderings import bw_from_adj
from data.mol_tokens import TOKENS_DICT_SEQ_MERGE_MOL, TOKENS_DICT_MOL, TOKENS_DICT_DIFF_MOL, TOKENS_DICT_DIFF_NI_MOL


PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
UNK_TOKEN = "<unk>"

standard_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

# TODO: fix moses / guacamol node_num and bw

dataset_list = ['GDSS_ego', 'GDSS_com', 'GDSS_enz', 'GDSS_grid', 'planar', 'sbm', 
                'proteins', 'lobster', 'point', 'ego', 'qm9', 'zinc', 'moses', 'guacamol']
# dataset_list = ['GDSS_ego']
# maximum number of nodes of each dataset (train, test, val)
node_num_list = [17, 20, 125, 361, 64, 187, 500, 98, 5037, 399, 9, 38, 31, 88]
bw_list = [15, 8, 19, 19, 26, 111, 125, 49, 167, 241, 5, 10, 12, 12]

TOKENS_DICT = {}
TOKENS_DICT_DIFF = {}
TOKENS_DICT_DIFF_NI = {}
TOKENS_DICT_DIFF_NI_REL = {}
TOKENS_DICT_FLATTEN = {}
TOKENS_DICT_SEQ = {}
TOKENS_BWR = {}

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
    
    tokens_seq = standard_tokens.copy()
    # 0: node token
    tokens_seq.append(0)
    # 1-n: edge relative position token
    tokens_seq.extend(np.arange(1,bw+1))
    TOKENS_DICT_SEQ[dataset] = tokens_seq
    # map adj_list_diff tokens
    tokens_diff = standard_tokens.copy()
    tokens_diff.extend([(num, b) for b in np.arange(1,bw+1) for num in np.arange(1,node_num) if (num-b >= 0)])
    TOKENS_DICT_DIFF[dataset] = tokens_diff
    # map bwr tokens
    # tokens_bwr = standard_tokens.copy()
    # for repeat in range(1,bw+1):
    #     tokens_bwr.extend(list(map(list, (product([0,1], repeat=repeat)))))
    # TOKENS_BWR[dataset] = tokens_bwr
    
    # map token_list_diff_ni tokens (src node: NI (0,1), tar node: difference)
    tokens_diff_ni = standard_tokens.copy()
    tokens_diff_ni.extend([(num, b) for b in np.arange(0,bw+1) for num in np.arange(0,2)])
    TOKENS_DICT_DIFF_NI[dataset] = tokens_diff_ni
    
    # map token_list_diff_ni_rel tokens (src node: NI (0,1, ..., BW), tar node: difference)
    tokens_diff_ni_rel = standard_tokens.copy()
    tokens_diff_ni_rel.extend([(num, b) for b in np.arange(1,bw+1) for num in np.arange(0,bw+1)])
    TOKENS_DICT_DIFF_NI_REL[dataset] = tokens_diff_ni_rel
    
    # map adj_flatten / adj_flatten_sym tokens
    tokens_flat = standard_tokens.copy()
    tokens_flat.extend([0, 1])
    TOKENS_DICT_FLATTEN[dataset] = tokens_flat

TOKENS_SPM_DICT = defaultdict()

for dataset in ['GDSS_com', 'GDSS_ego', 'planar', 'GDSS_enz', 'sbm', 'GDSS_grid']:
    # for string_type in ['adj_seq', 'adj_seq_rel', 'adj_flatten', 'adj_flatten_sym']:
    for string_type in ['adj_seq', 'adj_seq_rel']:
        for vocab_size in [200, 400]:
            key = f'{dataset}_{string_type}_{vocab_size}'
            TOKENS_SPM_DICT[key] = {}
            sp = spm.SentencePieceProcessor(model_file=f"resource/tokenizer/{dataset}/{string_type}_{vocab_size}.model")
            TOKENS_SPM_DICT[key]['sp'] = sp
            tokens_spm = [BOS_TOKEN, PAD_TOKEN, EOS_TOKEN]
            tokens_spm.extend([sp.IdToPiece(ids) for ids in range(sp.GetPieceSize())])
            TOKENS_SPM_DICT[key]['tokens'] = tokens_spm
            
for dataset in ['GDSS_com', 'GDSS_ego', 'planar', 'GDSS_enz', 'sbm']:
    # for string_type in ['adj_seq', 'adj_seq_rel', 'adj_flatten', 'adj_flatten_sym']:
    for string_type in ['adj_flatten', 'adj_flatten_sym']:
        for vocab_size in [200, 400]:
        # vocab_size = 200
            key = f'{dataset}_{string_type}_{vocab_size}'
            TOKENS_SPM_DICT[key] = {}
            sp = spm.SentencePieceProcessor(model_file=f"resource/tokenizer/{dataset}/{string_type}_{vocab_size}.model")
            TOKENS_SPM_DICT[key]['sp'] = sp
            tokens_spm = [BOS_TOKEN, PAD_TOKEN, EOS_TOKEN]
            tokens_spm.extend([sp.IdToPiece(ids) for ids in range(sp.GetPieceSize())])
            TOKENS_SPM_DICT[key]['tokens'] = tokens_spm
    
for dataset in ['GDSS_com', 'GDSS_ego', 'planar', 'GDSS_enz', 'sbm', 'lobster', 'ego', 'proteins']:
    for string_type in ['adj_seq_blank', 'adj_seq_rel_blank']:
        for vocab_size in [70]:
            key = f'{dataset}_{string_type}_{vocab_size}'
            TOKENS_SPM_DICT[key] = {}
            sp = spm.SentencePieceProcessor(model_file=f"resource/tokenizer/{dataset}/{string_type}_{vocab_size}.model")
            TOKENS_SPM_DICT[key]['sp'] = sp
            tokens_spm = [BOS_TOKEN, PAD_TOKEN, EOS_TOKEN]
            tokens_spm.extend([sp.IdToPiece(ids) for ids in range(sp.GetPieceSize())])
            TOKENS_SPM_DICT[key]['tokens'] = tokens_spm
            
# for dataset in ['planar', 'sbm', 'ego', 'proteins']:
#     for string_type in ['adj_seq_blank', 'adj_seq_rel_blank']:
#         for vocab_size in [1000]:
#             key = f'{dataset}_{string_type}_{vocab_size}'
#             TOKENS_SPM_DICT[key] = {}
#             sp = spm.SentencePieceProcessor(model_file=f"resource/tokenizer/{dataset}/{string_type}_{vocab_size}.model")
#             TOKENS_SPM_DICT[key]['sp'] = sp
#             tokens_spm = [BOS_TOKEN, PAD_TOKEN, EOS_TOKEN]
#             tokens_spm.extend([sp.IdToPiece(ids) for ids in range(sp.GetPieceSize())])
#             TOKENS_SPM_DICT[key]['tokens'] = tokens_spm

dataset = 'lobster'    
string_type = 'adj_flatten'
vocab_size = 200
key = f'{dataset}_{string_type}_{vocab_size}'
TOKENS_SPM_DICT[key] = {}
sp = spm.SentencePieceProcessor(model_file=f"resource/tokenizer/{dataset}/{string_type}_{vocab_size}.model")
TOKENS_SPM_DICT[key]['sp'] = sp
tokens_spm = [BOS_TOKEN, PAD_TOKEN, EOS_TOKEN]
tokens_spm.extend([sp.IdToPiece(ids) for ids in range(sp.GetPieceSize())])
TOKENS_SPM_DICT[key]['tokens'] = tokens_spm

dataset = 'GDSS_grid'    
for string_type in ['adj_seq_blank', 'adj_seq_rel_blank']:
    vocab_size = 65
    key = f'{dataset}_{string_type}_{vocab_size}'
    TOKENS_SPM_DICT[key] = {}
    sp = spm.SentencePieceProcessor(model_file=f"resource/tokenizer/{dataset}/{string_type}_{vocab_size}.model")
    TOKENS_SPM_DICT[key]['sp'] = sp
    tokens_spm = [BOS_TOKEN, PAD_TOKEN, EOS_TOKEN]
    tokens_spm.extend([sp.IdToPiece(ids) for ids in range(sp.GetPieceSize())])
    TOKENS_SPM_DICT[key]['tokens'] = tokens_spm
      
def token_list_to_dict(tokens):
    return {token: i for i, token in enumerate(tokens)}

TOKENS_KEY_DICT = {key: token_list_to_dict(value) for key, value in TOKENS_DICT.items()}
TOKENS_KEY_DICT_DIFF = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_DIFF.items()}
TOKENS_KEY_DICT_FLATTEN = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_FLATTEN.items()}
TOKENS_KEY_DICT_SEQ = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_SEQ.items()}
TOKENS_KEY_DICT_SPM = {key: token_list_to_dict(value['tokens']) for key, value in TOKENS_SPM_DICT.items()}
TOKENS_KEY_DICT_DIFF_NI = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_DIFF_NI.items()}
TOKENS_KEY_DICT_DIFF_NI_REL = {key: token_list_to_dict(value) for key, value in TOKENS_DICT_DIFF_NI_REL.items()}

def token_to_id(data_name, string_type, is_token=False, vocab_size=200, order='C-M'):
    if is_token or string_type in ['adj_seq_blank', 'adj_seq_rel_blank']:
        return TOKENS_KEY_DICT_SPM[f'{data_name}_{string_type}_{vocab_size}']
    elif string_type == 'adj_list':
        return TOKENS_KEY_DICT[data_name]
    elif string_type == 'adj_list_diff':
        return TOKENS_KEY_DICT_DIFF[data_name]
    elif string_type in ['adj_flatten', 'adj_flatten_sym', 'bwr']:
        return TOKENS_KEY_DICT_FLATTEN[data_name]
    elif string_type in ['adj_seq', 'adj_seq_rel']:
        return TOKENS_KEY_DICT_SEQ[data_name]
    elif string_type in ['adj_list_diff_ni']:
        # TODO: Fix here
        tokens = map_tokens(data_name, string_type, 0, order)
        return token_list_to_dict(tokens)
        # return TOKENS_KEY_DICT_DIFF_NI[data_name]
    elif string_type == 'adj_list_diff_ni_rel':
        tokens = map_tokens(data_name, string_type, 0, order)
        return token_list_to_dict(tokens)
    else:
        assert False, "No token type"

def id_to_token(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}

def tokenize(adj, adj_list, data_name, string_type, is_token=False, vocab_size=200, order='C-M'):
    TOKEN2ID = token_to_id(data_name, string_type, is_token, vocab_size, order)
    tokens = ["[bos]"]
    if is_token or string_type in ['adj_seq_blank', 'adj_seq_rel_blank']:
        key = f'{data_name}_{string_type}_{vocab_size}'
        sp = TOKENS_SPM_DICT[key]['sp']
        if string_type == 'adj_seq':
            string = map_string_adj_seq(adj_list)
        elif string_type == 'adj_seq_rel':
            string = map_string_adj_seq_rel(adj_list)
        elif string_type == 'adj_flatten':
            string = "".join([str(int(elem)) for elem in torch.flatten(torch.tensor(adj.todense())).tolist()])
        elif string_type == 'adj_flatten_sym':
            string = map_string_flat_sym(adj)
        elif string_type == 'adj_seq_rel_blank':
            string = map_string_adj_seq_rel_blank(adj_list)
        elif string_type == 'adj_seq_blank':
            string = map_string_adj_seq_blank(adj_list)
        
        tokens.extend(sp.encode_as_pieces(string))
    elif string_type == 'adj_list':
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
        prev_src_node = 0
        adj_list = sorted(adj_list, key=lambda x: (x[0], -x[1]))
        cur_tar_node = adj_list[0][1]
        for src_node, tar_node in adj_list:
            if prev_src_node != src_node:
                tokens.append(0)
                diff = src_node - tar_node
            else:
                diff = cur_tar_node - tar_node
            if diff != 0:
                tokens.append(diff)
            prev_src_node = src_node
            cur_tar_node = tar_node
    elif string_type in ['adj_list_diff_ni', 'adj_list_diff_ni_rel']:
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
    
    elif string_type == 'bwr':
        bw = bw_from_adj(adj.toarray())
        tokens.extend(torch.flatten(flatten_forward(torch.tensor(adj.todense()), bw)).tolist())
        
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
    if is_token or string_type in ['adj_seq_blank', 'adj_seq_rel_blank']:
        tokens = TOKENS_SPM_DICT[f'{data_name}_{string_type}_{vocab_size}']['tokens']
    elif string_type == 'adj_list':
        if data_name in ['qm9', 'zinc', 'moses', 'guacamol']:
            tokens = TOKENS_DICT_MOL[data_name]
        else:
            tokens = TOKENS_DICT[data_name]
    elif string_type == 'adj_list_diff':
        if data_name in ['qm9', 'zinc', 'moses', 'guacamol']:
            tokens = TOKENS_DICT_DIFF_MOL[data_name]
        else:
            tokens = TOKENS_DICT_DIFF[data_name]
    elif string_type == 'adj_list_diff_ni':
        if data_name in ['qm9', 'zinc', 'moses', 'guacamol']:
            tokens = TOKENS_DICT_DIFF_NI_MOL[data_name]
        else:
            # tokens = TOKENS_DICT_DIFF_NI[data_name]
            tokens = map_diff_ni_tokens(dataset=data_name, order=order)
    elif string_type == 'adj_list_diff_ni_rel':
        # tokens = TOKENS_DICT_DIFF_NI_REL[data_name]
        tokens = map_diff_ni_rel_tokens(dataset=data_name, order=order)
    elif string_type in ['adj_flatten', 'adj_flatten_sym', 'bwr']:
        tokens = TOKENS_DICT_FLATTEN[data_name]
    elif string_type in ['adj_seq', 'adj_seq_rel']:
        tokens = TOKENS_DICT_SEQ[data_name]
    elif string_type in ['adj_seq_merge', 'adj_seq_rel_merge']:
        tokens = TOKENS_DICT_SEQ_MERGE_MOL[data_name]

    else:
        assert False, "No token type"
    return tokens

def check_reconstruction(data_name, string_type, order):
    adj_list = [(1,0), (2,0), (5,2), (5,3), (6,0), (7,4)]
    tokens = map_tokens(data_name, string_type, vocab_size=None, order=order, is_token=False)
    sequence = tokenize(None, adj_list, data_name, string_type)
    ID2TOKEN = id_to_token(tokens)
    print([ID2TOKEN[i] for i in sequence])
    un_tokens = untokenize(sequence, data_name, string_type, is_token=False, order=order)[0]
    # TODO: change the mapping function dependent on string_type
    a = adj_list_diff_ni_to_adj_list(un_tokens)
    print(a)
    print(a)