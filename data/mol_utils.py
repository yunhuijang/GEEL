import networkx as nx
from rdkit import Chem
from tqdm import tqdm
import torch
import numpy as np
import re

from data.data_utils import adj_list_diff_ni_to_adj_list
from data.mol_tokens import BOND_TYPE_DICT, ATOMFEAT2TOKEN, TOKEN2ATOMFEAT


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
    token_set = set()
    for mol in tqdm(mols, 'Molecules to graph'):
        if not mol:
            continue
        G = nx.Graph()

        for atom in mol.GetAtoms():
            token = get_atom_token(atom)
            token_set.add(token)
            G.add_node(atom.GetIdx(), token=token)
        
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=BOND_TYPE_DICT[bond.GetBondTypeAsDouble()])
        nx_graphs.append(G)
        
    return nx_graphs, token_set

ATOM_VALENCY = {'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 3, 'S': 2, 'Cl': 1, 'Br': 1, 'I': 1, 'B': 3, 'Se': 6, 'Si': 4}

bond_decoder = {5: Chem.rdchem.BondType.SINGLE, 6: Chem.rdchem.BondType.DOUBLE, 
                7: Chem.rdchem.BondType.TRIPLE, 8: Chem.rdchem.BondType.AROMATIC}
NODE_TYPE_TO_ATOM_NUM = {9: 9, 10: 8, 11: 7, 12: 6, 13: 15, 14: 53, 15: 17, 16: 35, 17: 16, 18:5, 19:34, 20:14}

def generate_mol(x, adj):
    mol = construct_mol(x, adj)
    cmol, no_correct = correct_mol(mol)
    vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=True)
    return vcmol, no_correct
    
def construct_mol(x, adj):
    mol = Chem.RWMol()
    for atom in x:
        # mol.AddAtom(Chem.Atom(NODE_TYPE_TO_ATOM_NUM[atom]))
        atomic_num, formal_charge, num_explicit_Hs = TOKEN2ATOMFEAT[atom]
        atom = Chem.Atom(atomic_num)
        atom.SetFormalCharge(formal_charge)
        atom.SetNumExplicitHs(num_explicit_Hs)
        if num_explicit_Hs > 0:
            atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    
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
                an_symbol = mol.GetAtomWithIdx(idx).GetSymbol()
                # if an in (10, 11, 17) and (v - ATOM_VALENCY[an]) == 1:
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an_symbol]) == 1:
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

def get_edge_from_adj_list(sample):
    return [elem for elem in sample if type(elem) is tuple]

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

def map_featured_samples_to_adjs(samples, string_type='adj_list_diff_ni'):
    if string_type != 'adj_list_diff_ni':
        raise ValueError('String type must be adj_list_diff_ni for molecules')
    # return weighted adjacency matrix (edge features) and node features
    samples = [sample for sample in samples if len(sample)>0]
    samples = [sample for sample in samples if check_diff_ni_validity_mol(sample)]
    feature_seqs = [get_feature_from_adj_list_diff_ni(sample) for sample in samples]

    adj_lists = [get_edge_from_adj_list(sample) for sample in samples]
        
    adj_lists = [adj_list_diff_ni_to_adj_list(adj_list) for adj_list in adj_lists]
    xs = [[elem for elem in feature_seq if elem in TOKEN2ATOMFEAT.keys()] for feature_seq in feature_seqs]
    edge_features = [[elem for elem in feature_seq if elem in bond_decoder.keys()] for feature_seq in feature_seqs]

    featured_adj_lists = [map_featured_adj_list(adj_list, edge_feature) for adj_list, edge_feature in zip(adj_lists, edge_features)]
    weighted_adjs = [featured_adj_list_to_adj(featured_adj_list) for featured_adj_list in featured_adj_lists]
    for adj in weighted_adjs:
        np.fill_diagonal(adj, 0)
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

def get_atom_token(atom):
    return ATOMFEAT2TOKEN[(atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetNumExplicitHs())]