import networkx as nx
from rdkit import Chem
from tqdm import tqdm
import torch
import numpy as np
import re

from data.data_utils import seq_to_adj, seq_rel_to_adj, seq_to_adj_list, seq_rel_to_adj_list, adj_list_diff_to_adj_list, adj_list_diff_ni_to_adj_list
from data.mol_tokens import NODE_TYPE_DICT, BOND_TYPE_DICT


DATA_DIR = "resource"

# codes adapted from https://github.com/harryjo97/GDSS

def canonicalize_smiles(smiles):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in tqdm(smiles, 'Canonicalizing')]

def mols_to_smiles(mols):
    return [Chem.MolToSmiles(mol) for mol in tqdm(mols, 'molecules to SMILES')]

def smiles_to_mols(smiles):
    return [Chem.MolFromSmiles(s) for s in tqdm(smiles, 'SMILES to molecules')]

def add_self_loop(graph):
    for node in graph.nodes:
        node_label = graph.nodes[node]['label']
        graph.add_edge(node, node, label=node_label)
    return graph

def mols_to_nx(mols):
    nx_graphs = []
    for mol in tqdm(mols, 'Molecules to graph'):
        if not mol:
            continue
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=NODE_TYPE_DICT[atom.GetSymbol()])
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=BOND_TYPE_DICT[bond.GetBondTypeAsDouble()])
        nx_graphs.append(G)
        
    return nx_graphs

def check_adj_validity_mol(adj):
    if adj.size == 0:
        return None
    non_padded_index = max(max(np.argwhere(adj.any(axis=0)))[0], max(np.argwhere(adj.any(axis=1)))[0])+1
    x = adj.diagonal()[:non_padded_index]
    # check if diagonal elements are all node types and all diagonal elements are full / not proper bond type
    if len([atom for atom in x if atom in NODE_TYPE_DICT.values()]) == non_padded_index:
        # not proper bond type
        check_bond_adj = adj.copy()
        np.fill_diagonal(check_bond_adj, 0)
        bond_type_set = set(check_bond_adj.flatten())
        bond_type_set.remove(0)
        if len([bt for bt in bond_type_set if bt not in BOND_TYPE_DICT.values()]) == 0:
            return adj
        else:
            return None
    else:
        return None

ATOM_VALENCY = {12: 4, 11: 3, 10: 2, 9: 1, 13: 3, 17: 2, 15: 1, 16: 1, 14: 1, 18:3, 19:2, 20:4}
bond_decoder = {5: Chem.rdchem.BondType.SINGLE, 6: Chem.rdchem.BondType.DOUBLE, 
                7: Chem.rdchem.BondType.TRIPLE, 8: Chem.rdchem.BondType.AROMATIC}
NODE_TYPE_TO_ATOM_NUM = {9: 9, 10: 8, 11: 7, 12: 6, 13: 15, 14: 53, 15: 17, 16: 35, 17: 16}

def generate_mol(x, adj):
    mol = construct_mol(x, adj)
    cmol, no_correct = correct_mol(mol)
    vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=True)
    return vcmol, no_correct
    
def construct_mol(x, adj):
    mol = Chem.RWMol()
    for atom in x:
        mol.AddAtom(Chem.Atom(NODE_TYPE_TO_ATOM_NUM[atom]))
    
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder[adj[start, end]])
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (10, 11, 17) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence

# codes adapted from https://github.com/cvignac/DiGress
def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type = int(b.GetBondType())
                queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                if type == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder[t+4])
    return mol, no_correct

def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol

def fix_symmetry_mol(weighted_adj):
    sym_adj = torch.tril(weighted_adj) + torch.tril(weighted_adj).T
    sym_adj[range(len(sym_adj)), range(len(sym_adj))] = sym_adj.diagonal()/2
    return sym_adj

def check_adj_feature_seq_size(adj_list, feature_list):
    # check the validity with the length of adj_seq and feature_seq
    if len(adj_list) + 1 == len(feature_list):
        return True
    else:
        return False

def check_adj_feature_seq_validity(adj_seq, string_type):
    if len(adj_seq) == 0:
        return False
    if adj_seq[0] != 0:
        return False
    if string_type in ['adj_seq_merge', 'adj_seq_rel_merge']:
        # remove the first node (same as adj_seq / adj_seq_rel)
        adj_seq = adj_seq[1:]
        if len(adj_seq) == 0:
            return False
        if adj_seq[0] != 0:
            return False
    if string_type in ['adj_seq_merge', 'adj_seq']:
        if len(seq_to_adj(adj_seq)) == 0:
            return False
    if string_type in ['adj_seq_rel_merge', 'adj_seq_rel']:
        if len(seq_rel_to_adj(adj_seq)) == 0:
            return False
    return True

def get_element_list_from_tuple(list_of_tuples, elem_num=0):
    return [tup[elem_num] for tup in list_of_tuples]

def get_edge_from_adj_list(sample):
    return [elem for elem in sample if type(elem) is tuple]

def get_feature_from_adj_list(sample):
    return [elem for elem in sample if type(elem) is not tuple]

def get_feature_from_adj_list_diff_ni(sample):
    # filter out self loop bond features
    sample_tuple_list = [(prev, cur) for prev, cur in zip(sample[:-1], sample[1:])]
    feature_list = [sample[0]]
    feature_list.extend([cur for prev, cur in sample_tuple_list if (type(cur) is not tuple) and (prev != (1,0))])
    return feature_list

def check_diff_ni_validity_mol(sample):
    adj_list = get_edge_from_adj_list(sample)
    if adj_list[0][0] == 0:
        return False
    else:
        return True

def map_featured_samples_to_adjs(samples, samples_feature, string_type):
    # return weighted adjacency matrix (edge features) and node features
    if string_type in ['adj_seq', 'adj_seq_rel']:
        valid_adj_feature_seqs = [(adj_list, feature_list) for adj_list, feature_list in zip(samples, samples_feature)
                                  if (check_adj_feature_seq_size(adj_list, feature_list)) and check_adj_feature_seq_validity(adj_list, string_type)]
        adj_seqs = [adj_feature_list[0] for adj_feature_list in valid_adj_feature_seqs]
        feature_seqs = [adj_feature_list[1] for adj_feature_list in valid_adj_feature_seqs]
    elif string_type in ['adj_seq_merge', 'adj_seq_rel_merge']:
        valid_adj_feature_seqs = [sample for sample in samples if check_adj_feature_seq_validity(get_element_list_from_tuple(sample, 0), string_type)]
        adj_seqs = [get_element_list_from_tuple(adj_feature_list, 0)[1:] for adj_feature_list in valid_adj_feature_seqs]
        feature_seqs = [get_element_list_from_tuple(adj_feature_list, 1) for adj_feature_list in valid_adj_feature_seqs]
    elif string_type in ['adj_list', 'adj_list_diff', 'adj_list_diff_ni']:
        samples = [sample for sample in samples if len(sample)>0]
        if string_type == 'adj_list_diff_ni':
            samples = [sample for sample in samples if check_diff_ni_validity_mol(sample)]
            feature_seqs = [get_feature_from_adj_list_diff_ni(sample) for sample in samples]
        else:
            feature_seqs = [get_feature_from_adj_list(sample) for sample in samples]
        adj_lists = [get_edge_from_adj_list(sample) for sample in samples]
            
        if string_type == 'adj_list_diff':
            adj_lists = [adj_list_diff_to_adj_list(adj_list) for adj_list in adj_lists]
        elif string_type == 'adj_list_diff_ni':
            adj_lists = [adj_list_diff_ni_to_adj_list(adj_list) for adj_list in adj_lists]
    if 'adj_seq' in string_type:
        if string_type in ['adj_seq', 'adj_seq_merge']:
            adj_lists = [seq_to_adj_list(seq) for seq in adj_seqs if len(seq_to_adj(seq))>0]
        elif string_type in ['adj_seq_rel', 'adj_seq_rel_merge']:
            adj_lists = [seq_rel_to_adj_list(seq_rel) for seq_rel in adj_seqs if len(seq_rel_to_adj(seq_rel))>0]
        # map node features
        node_indices = [np.where(np.array(adj_seq)==0)[0] for adj_seq in adj_seqs]
        xs = [[feature_seq[0]] for feature_seq in feature_seqs]
        for x, feature_seq, node_index in zip(xs, feature_seqs, node_indices):
            x.extend([feature_seq[i+1] for i in node_index])
        # map weighted adjacency matrices
        edge_indices = [np.where(np.array(adj_seq)!=0)[0] for adj_seq in adj_seqs]
        edge_features = [[feature_seq[i+1] for i in edge_index] for feature_seq, edge_index in zip(feature_seqs, edge_indices)]
    else:
        xs = [[elem for elem in feature_seq if elem in NODE_TYPE_TO_ATOM_NUM.keys()] for feature_seq in feature_seqs]
        if string_type == 'adj_list_diff_ni':
            edge_features = [[elem for elem in feature_seq if elem in bond_decoder.keys()] for feature_seq in feature_seqs]
        else:
            edge_features = [[elem for elem in feature_seq if elem in bond_decoder.keys()] for feature_seq in feature_seqs]
        
    featured_adj_lists = [map_featured_adj_list(adj_list, edge_feature) for adj_list, edge_feature in zip(adj_lists, edge_features)]
    weighted_adjs = [featured_adj_list_to_adj(featured_adj_list) for featured_adj_list in featured_adj_lists]
    if string_type == 'adj_list_diff_ni':
        for adj in weighted_adjs:
            np.fill_diagonal(adj, 0)
        # weighted_adjs = [np.fill_diagonal(adj, 0) for adj in weighted_adjs]
    final_weighted_adjs = [weighted_adj for weighted_adj, x in zip(weighted_adjs, xs) if (len(weighted_adj) == len(x)) and (len(weighted_adj)>1)]
    final_xs = [x for weighted_adj, x in zip(weighted_adjs, xs) if (len(weighted_adj) == len(x)) and (len(weighted_adj)>1)]

    return final_weighted_adjs, final_xs
            
def adj_x_to_graph_mol(weighted_adj, x, is_cuda=False):
    if is_cuda:
        weighted_adj = weighted_adj.detach().cpu().numpy()
    
    non_padded_index = max(max(np.argwhere(weighted_adj.any(axis=0)))[0], max(np.argwhere(weighted_adj.any(axis=1)))[0])+1
    adj = weighted_adj[:non_padded_index, :non_padded_index]
    
    mol, no_correct = generate_mol(x, adj)
    return mol, no_correct

def map_featured_adj_list(adj_list, edge_feature):
    # Add edge features to adj_list (triplet: (src node, tar node, edge feature))
    featured_adj_list = []
    for edge, edge_feature in zip(adj_list, edge_feature):
        featured_adj_list.append(edge + (edge_feature,)) 
        
    return featured_adj_list

def make_empty_adj(num_nodes):
    adj = [[0] * num_nodes for _ in range(num_nodes)]
    return np.array(adj)
    

def featured_adj_list_to_adj(adj_list):
    '''
    edge featured adjacency list to weighted adjacency matrix
    '''
    if len(adj_list) < 2:
        return make_empty_adj(1)
    
    max_src_node = max([elem[0] for elem in adj_list])
    max_tar_node = max([elem[1] for elem in adj_list])
    max_num_nodes = max(max_src_node, max_tar_node)+1
    
    if max_num_nodes < 2:
        return make_empty_adj(1)
    
    adj = [[0] * max_num_nodes for _ in range(max_num_nodes)]
    
    for n, e, f in adj_list:
        adj[n][e] = f
        adj[e][n] = f

    return np.array(adj)