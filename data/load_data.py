import pickle
import networkx as nx

def get_community_small_data():
    try:
        with open('gcg/resource/community_small.pkl', 'rb') as f:
            graph_list = pickle.load(f)
    except FileNotFoundError:
        with open('resource/community_small.pkl', 'rb') as f:
            graph_list = pickle.load(f)
    graphs = [nx.convert_node_labels_to_integers(graph) for graph in graph_list]
    return graphs

def get_ego_small_data():
    try:
        with open('gcg/resource/ego_small.pkl', 'rb') as f:
            graph_list = pickle.load(f)
    except FileNotFoundError:
        with open('resource/ego_small.pkl', 'rb') as f:
            graph_list = pickle.load(f)
    graphs = [nx.convert_node_labels_to_integers(graph) for graph in graph_list]
    return graphs

def get_gdss_grid_data():
    with open('resource/grid.pkl', 'rb') as f:
        graph_list = pickle.load(f)
    graphs = [nx.convert_node_labels_to_integers(graph) for graph in graph_list]
    return graphs

def get_gdss_enzymes_data():
    with open('resource/ENZYMES.pkl', 'rb') as f:
        graph_list = pickle.load(f)
    graphs = [nx.convert_node_labels_to_integers(graph) for graph in graph_list]
    return graphs

def get_grid_small_data():
    with open('resource/grid_small.pkl', 'rb') as f:
        graph_list = pickle.load(f)
    graphs = [nx.convert_node_labels_to_integers(graph) for graph in graph_list]
    return graphs

def get_caveman_data():
    with open('resource/caveman.pkl', 'rb') as f:
        graph_list = pickle.load(f)
    graphs = [nx.convert_node_labels_to_integers(graph) for graph in graph_list]
    return graphs