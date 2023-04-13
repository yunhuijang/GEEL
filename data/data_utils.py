import torch
from torch.nn import ZeroPad2d
from torch import LongTensor
from torch import count_nonzero
import math
from collections import deque
from time import time
from treelib import Tree, Node
from sklearn.model_selection import train_test_split
from itertools import zip_longest
import networkx as nx
from joblib import Parallel, delayed


def get_level(node):
    return int(node.identifier.split('-')[0])

def get_location(node):
    return int(node.identifier.split('-')[1])

def get_k(tree):
    return math.sqrt(len(tree[tree.root].successors(tree.identifier)))

def get_parent(node, tree):
    return tree[node.predecessor(tree.identifier)]

def get_children_identifier(node, tree):
    return node.successors(tree.identifier)

def nearest_power_2(N):
    a = int(math.log2(N)) 
    if 2**a == N:
        return N

    return 2**(a + 1)

def adj_to_k2_tree(adj, return_tree=False, is_wholetree=False, k=4):
    # TODO: generalize for other k
    n_org_nodes = adj.shape[0]
    # add padding (proper size for k)
    n_nodes = nearest_power_2(n_org_nodes)
    
    padder = ZeroPad2d((0, n_nodes-n_org_nodes, 0, n_nodes-n_org_nodes))
    padded_adj = padder(adj)
    k_sqrt = math.sqrt(k)
    total_level = int(math.log(n_nodes, k_sqrt))
    tree_list = []
    leaf_list = []
    tree = Tree()
    # add root node
    tree.create_node("root", "0")
    tree_key_list = deque([])
    slice_size = int(n_nodes / k_sqrt)
    # slice matrices 
    sliced_adjs = deque([padded_adj[:slice_size, :slice_size], padded_adj[:slice_size, slice_size:], 
            padded_adj[slice_size:, :slice_size], padded_adj[slice_size:, slice_size:]])
    sliced_adjs_is_zero = LongTensor([int(count_nonzero(adj)>0) for adj in sliced_adjs])
    tree_list.append(sliced_adjs_is_zero)
    tree_element_list = deque(sliced_adjs_is_zero)
    for i, elem in enumerate(tree_element_list, 1):
        tree.create_node(elem, f"1-{i}", parent="0")
        tree_key_list.append(f"1-{i}")
    
    while (slice_size != 1):
        n_nodes = sliced_adjs[0].shape[0]
        if n_nodes == 2:
            if is_wholetree:
                leaf_list = [adj.reshape(4,) for adj in sliced_adjs]
            else:
                leaf_list = [adj.reshape(4,) for adj in sliced_adjs if count_nonzero(adj)>0]
            break
        slice_size = int(n_nodes / k_sqrt)
        target_adj = sliced_adjs.popleft()
        target_adj_size = target_adj.shape[0]
        if return_tree:
            parent_node_key = tree_key_list.popleft()
        # remove adding leaves to 0
        if not is_wholetree:
            if count_nonzero(target_adj) == 0:
                continue
        # generate tree_list and leaf_list
        new_sliced_adjs = [target_adj[:slice_size, :slice_size], target_adj[:slice_size, slice_size:], 
                target_adj[slice_size:, :slice_size], target_adj[slice_size:, slice_size:]]
        new_sliced_adjs_is_zero = LongTensor([int(count_nonzero(adj)>0) for adj in new_sliced_adjs])
        sliced_adjs.extend(new_sliced_adjs)
        tree_list.append(new_sliced_adjs_is_zero)
        
        if return_tree:
            # generate tree
            tree_element_list.extend(new_sliced_adjs_is_zero)
            cur_level = int(total_level - math.log(target_adj_size, k_sqrt) + 1)
            cur_level_key_list = [int(key.split('-')[1]) for key in tree_key_list if int(key.split('-')[0]) == cur_level]
            if len(cur_level_key_list) > 0:
                key_starting_point = max(cur_level_key_list)
            else:
                key_starting_point = 0
            for i, elem in enumerate(new_sliced_adjs_is_zero, key_starting_point+1):
                tree.create_node(elem, f"{cur_level}-{i}", parent=parent_node_key)
                tree_key_list.append(f"{cur_level}-{i}")
            
    if return_tree:
        # add leaves to tree
        leaves = [node for node in tree.leaves() if node.tag == 1]
        index = 1
        for leaf, leaf_values in zip(leaves, leaf_list):
            for value in leaf_values:
                tree.create_node(int(value), f"{total_level}-{index}", parent=leaf)
                index += 1
        return tree
    else:
        return tree_list, leaf_list


def tree_to_adj(tree):
    '''
    convert k2 tree to adjacency matrix
    '''
    tree = map_starting_point(tree)
    k = get_k(tree)
    depth = tree.depth()
    leaves = [leaf for leaf in tree.leaves() if leaf.tag == 1]
    one_data_points = [leaf.data for leaf in leaves]
    x_list = [data[0] for data in one_data_points]
    y_list = [data[1] for data in one_data_points]
    matrix_size = int(k**depth)
    adj = torch.zeros((matrix_size, matrix_size))
    adj[x_list, y_list] = 1
    
    return adj
    
def map_starting_point(tree):
    '''
    map starting points for each elements in tree (to convert adjacency matrix)
    '''
    bfs_list = [tree[node] for node in tree.expand_tree(mode=Tree.WIDTH, 
                                                            key=lambda x: (int(x.identifier.split('-')[0]), int(x.identifier.split('-')[1])),)]
    for node in bfs_list:
        if node.is_root():
            node.data = (0,0)
        else:
            parent = get_parent(node, tree)
            siblings = get_children_identifier(parent, tree)
            index = siblings.index(node.identifier)
            level = get_level(node)
            tree_depth = tree.depth()
            k = get_k(tree)
            matrix_size = k**tree_depth
            adding_value = int(matrix_size/(k**level))
            parent_starting_point = parent.data
            if index == 0:
                node.data = parent_starting_point
            elif index == 1:
                node.data = (parent_starting_point[0], parent_starting_point[1]+adding_value)
            elif index == 2:
                node.data = (parent_starting_point[0]+adding_value, parent_starting_point[1])
            else:
                node.data = (parent_starting_point[0]+adding_value, parent_starting_point[1]+adding_value)
            
    return tree

def tree_to_dfs_string(tree):
    '''
    convert k2 tree into dfs string representation
    '''
    dfs_node_list = [tree[node] for node in tree.expand_tree(mode=Tree.DEPTH, 
                                                            key=lambda x: (int(x.identifier.split('-')[0]), int(x.identifier.split('-')[1])))][1:]
    string = ''
    # initialization
    first_value = int(dfs_node_list[0].tag)
    string += str(first_value)
    for prev, cur in zip(dfs_node_list, dfs_node_list[1:]):
        prev_key = get_level(prev)
        cur_key = get_level(cur)
        cur_value = int(cur.tag)
        if prev_key < cur_key:
            string += f'({cur_value}'
            continue
        elif prev_key == cur_key:
            string += f'{cur_value}'
            continue
        else:
            while(prev_key - cur_key > 0):
                string += ')'
                prev_key -= 1
            string += f'{cur_value}'
            continue
    string += ')'
    return string

def map_child_deg(node, tree):
    '''
    return sum of direct children nodes' degree (tag)
    '''
    if node.is_leaf():
        return str(int(node.tag))
    
    children = get_children_identifier(node, tree)
    child_deg = sum([int(tree[child].tag) for child in children])
    
    return str(child_deg)

def tree_to_bfs_string(tree, string_type='bfs'):
    bfs_node_list = [tree[node] for node in tree.expand_tree(mode=tree.WIDTH,
                                                             key=lambda x: (int(x.identifier.split('-')[0]), int(x.identifier.split('-')[1])))][1:]
    if string_type in ['bfs', 'group']:
        bfs_value_list = [str(int(node.tag)) for node in bfs_node_list]
    elif string_type == 'bfs-deg':
        bfs_value_list = Parallel(n_jobs=8)(delayed(map_child_deg)(node, tree) for node in bfs_node_list)
    
    return ''.join(bfs_value_list)

def dfs_string_to_tree(string):
    tree = Tree()
    tree.create_node("root", "0")
    parent_node = tree["0"]
    string = iter(string)
    # initialization
    char = next(string, None)
    tree.create_node(int(char), "1-1", parent=parent_node)
    parent_node = tree["1-1"]
    while True:
        char = next(string, None)
        if char == None:
            break
        elif char == '(':
            char = next(string)
        elif char == ')':
            try:
                char = next(string)
            except StopIteration:
                break
            parent_node = get_parent(get_parent(parent_node, tree), tree)
        else:
            parent_node = get_parent(parent_node, tree)
            
        if char in ['0', '1']:
            parent_level = get_level(parent_node)
            cur_level_max = max([get_location(node) for node in tree.nodes.values() if get_level(node) == parent_level+1], default=0)
            new_node_key = f"{parent_level+1}-{cur_level_max+1}"
            tree.create_node(int(char), new_node_key, parent=parent_node)
            parent_node = tree[new_node_key]
    return tree

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def bfs_string_to_tree(string):
    tree = Tree()
    tree.create_node("root", "0")
    parent_node = tree["0"]
    node_deque = deque([])
    for node_1, node_2, node_3, node_4 in grouper(4, string):
        parent_level = get_level(parent_node)
        cur_level_max = max([get_location(node) for node in tree.nodes.values() if get_level(node) == parent_level+1], default=0)
        for i, node_tag in enumerate([node_1, node_2, node_3, node_4], 1):
            if node_tag == None:
                break
            new_node = Node(tag=int(node_tag), identifier=f"{parent_level+1}-{cur_level_max+i}")
            tree.add_node(new_node, parent=parent_node)
            node_deque.append(new_node)
        parent_node = node_deque.popleft()
        while(parent_node.tag == 0):
            if len(node_deque) == 0:
                return tree
            parent_node = node_deque.popleft()
    return tree

def clean_string(string):

    
    if "[pad]" in string:
        string = string[:string.index("[pad]")]
        
    return string

def check_validity(string):
    '''
    check validity of DFS string ("closed parenthesis")
    '''
    if len(string) == 0:
        return False
    if string[0] in [')', '(']:
        return False
    
    stack = []
    for i in string:
        if i == '(':
            stack.append(i)
        elif i == ')':
            if ((len(stack) > 0) and
                ('(' == stack[-1])):
                stack.pop()
            else:
                return False
    if len(stack) == 0:
        return True
    else:
        return False
    
def train_val_test_split(
    data: list,
    train_size: float = 0.7, val_size: float = 0.1, test_size: float = 0.2,
    seed: int = 42,
):
    train_val, test = train_test_split(data, train_size=train_size + test_size, random_state=seed)
    train, val = train_test_split(train_val, train_size=train_size / (train_size + val_size), random_state=seed)
    return train, val, test

def adj_to_graph(adj, is_cuda=False):
    if is_cuda:
        adj = adj.detach().cpu().numpy()
    G = nx.from_numpy_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() < 1:
        G.add_node(1)
    return G

def map_tree_pe(tree):
    depth = tree.depth()
    k = get_k(tree)
    size = int(depth*(k**2))
    pe = torch.zeros((size))
    for node in tree.nodes.values():
        node.data = pe
        if not node.is_root():
            parent = get_parent(node, tree)
            branch = get_children_identifier(parent, tree).index(node.identifier)
            current_pe = torch.zeros(int(k**2))
            current_pe[branch] = 1
            pe = torch.cat((current_pe, parent.data[:int(size-k**2)]))
            node.data = pe
    return tree