import networkx as nx
import pickle
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import torch
import numpy as np
import re

from data.orderings import ORDER_FUNCS, order_graphs
from data.data_utils import train_val_test_split, adj_to_k2_tree, map_child_deg, TYPE_NODE_DICT, NODE_TYPE_DICT, BOND_TYPE_DICT


DATA_DIR = "resource"

def canonicalize_smiles(smiles):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in tqdm(smiles, 'Canonicalizing')]

def mols_to_smiles(mols):
    return [Chem.MolToSmiles(mol) for mol in tqdm(mols, 'molecules to SMILES')]

def smiles_to_mols(smiles):
    return [Chem.MolFromSmiles(s) for s in tqdm(smiles, 'SMILES to molecules')]

def tree_to_bfs_string_mol(tree, string_type='bfs'):
    bfs_node_list = [tree[node] for node in tree.expand_tree(mode=tree.WIDTH,
                                                             key=lambda x: (int(x.identifier.split('-')[0]), int(x.identifier.split('-')[1])))][1:]
    if string_type in ['bfs', 'group', 'bfs-tri']:
        bfs_value_list = [str(int(node.tag)) for node in bfs_node_list]
        if string_type == 'bfs-tri':
            bfs_value_list = [bfs_value_list[i] for i in range(len(bfs_value_list)) if i % 4 !=2]
    elif string_type in ['bfs-deg', 'bfs-deg-group']:
        bfs_value_list = [map_child_deg(node, tree) for node in bfs_node_list]
    
    final_value_list = [TYPE_NODE_DICT[token] if token in TYPE_NODE_DICT.keys() else token for token in bfs_value_list]
    
    return ''.join(final_value_list)

def add_self_loop(graph):
    for node in graph.nodes:
        node_label = graph.nodes[node]['label']
        graph.add_edge(node, node, label=node_label)
    return graph

def map_new_ordered_graph(ordered_graph):
    '''
    Map ordered_graph object to ordered networkx graph
    '''
    org_graph = ordered_graph.graph
    ordering = ordered_graph.ordering
    mapping = {i: ordering.index(i) for i in range(len(ordering))}
    new_graph = nx.relabel_nodes(org_graph, mapping)
    return new_graph


def generate_mol_string(dataset_name, order='C-M', is_small=False):
    '''
    Generate strings for each dataset / split (without degree (only 0-1))
    '''
    # load molecule graphs
    col_dict = {'qm9': 'SMILES1', 'zinc': 'smiles'}
    df = pd.read_csv(f'{DATA_DIR}/{dataset_name}/{dataset_name}.csv')
    smiles = list(df[col_dict[dataset_name]])
    if is_small:
        smiles = smiles[:100]
    smiles = [s for s in smiles if len(s)>1]
    smiles = canonicalize_smiles(smiles)
    split = ['train', 'val', 'test']
    train_smiles, val_smiles, test_smiles = train_val_test_split(smiles)
    for s, split in zip([train_smiles, val_smiles, test_smiles], split):
        with open(f'{DATA_DIR}/{dataset_name}/{dataset_name}_smiles_{split}.txt', 'w') as f:
            for string in s:
                f.write(f'{string}\n')
    graph_list = []
    for smiles in train_smiles, val_smiles, test_smiles:
        mols = smiles_to_mols(smiles)
        graphs = mols_to_nx(mols)
        graphs = [add_self_loop(graph) for graph in tqdm(graphs, 'Adding self-loops')]
        num_rep = 1
        # order graphs
        order_func = ORDER_FUNCS[order]
        total_graphs = graphs
        total_ordered_graphs = order_graphs(total_graphs, num_repetitions=num_rep, order_func=order_func, seed=0, is_mol=True)
        new_ordered_graphs = [map_new_ordered_graph(graph) for graph in tqdm(total_ordered_graphs, 'Map new ordered graphs')]
        graph_list.append(new_ordered_graphs)
    
    # write graphs
    
    
    for graphs, split in zip(graph_list, split):
        weighted_adjs = [nx.attr_matrix(graph, edge_attr='label', rc_order=range(len(graph))) for graph in graphs]
        trees = [adj_to_k2_tree(torch.Tensor(adj), return_tree=True, is_mol=True) for adj in tqdm(weighted_adjs, 'Generating tree from adj')]
        strings = [tree_to_bfs_string_mol(tree, string_type='bfs-deg-group') for tree in tqdm(trees, 'Generating strings from tree')]
        if is_small:
            file_name = f'{dataset_name}_small_str_{split}'
        else:
            file_name = f'{dataset_name}_str_{split}'
        with open(f'{DATA_DIR}/{dataset_name}/{file_name}.txt', 'w') as f:
            for string in strings:
                f.write(f'{string}\n')
        if split == 'test':
            with open(f'{DATA_DIR}/{dataset_name}/{dataset_name}_test_graphs.pkl', 'wb') as f:
                pickle.dump(graphs, f)

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
    # np.fill_diagonal(check_adj, 0)
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

def adj_to_graph_mol(weighted_adj, is_cuda=False):
    if is_cuda:
        weighted_adj = weighted_adj.detach().cpu().numpy()
    
    non_padded_index = max(max(np.argwhere(weighted_adj.any(axis=0)))[0], max(np.argwhere(weighted_adj.any(axis=1)))[0])+1
    adj = weighted_adj[:non_padded_index, :non_padded_index]
    
    x = adj.diagonal().copy()
    np.fill_diagonal(adj, 0)
    
    mol, no_correct = generate_mol(x, adj)
    return mol, no_correct

ATOM_VALENCY = {12: 4, 11: 3, 10: 2, 9: 1, 13: 3, 17: 2, 15: 1, 16: 1, 14: 1}
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
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
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