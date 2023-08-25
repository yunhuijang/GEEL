import torch
import torch.nn.functional as F
import os
import pickle
import numpy as np
from scipy.linalg import toeplitz
import pyemd
import concurrent.futures
from datetime import datetime
from scipy.linalg import eigvalsh
import networkx as nx
from functools import partial
import random
import subprocess as sp
from eden.graph import vectorize
from sklearn.metrics.pairwise import pairwise_kernels
import json

from data.tokens import TOKENS_DICT, TOKENS_DICT_DIFF, TOKENS_DICT_FLATTEN, TOKENS_DICT_SEQ, TOKENS_SPM_DICT
from data.mol_tokens import TOKENS_DICT_SEQ_MOL, TOKENS_DICT_FLATTEN_MOL, TOKENS_DICT_MOL
from data.data_utils import load_graphs
from evaluation.evaluation_spectre import eval_acc_grid_graph, eval_acc_planar_graph, eval_acc_sbm_graph

def save_graph_list(log_folder_name, exp_name, gen_graph_list):
    if not(os.path.isdir(f'./samples/graphs/{log_folder_name}')):
        os.makedirs(os.path.join(f'./samples/graphs/{log_folder_name}'))
    if not(os.path.isdir(f'./samples/string/{log_folder_name}')):
        os.makedirs(os.path.join(f'./samples/string/{log_folder_name}'))
    with open(f'./samples/graphs/{log_folder_name}/{exp_name}.pkl', 'wb') as f:
            pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    save_dir = f'./samples/graphs/{log_folder_name}/{exp_name}.pkl'
    return save_dir

def compute_sequence_accuracy(logits, batched_sequence_data, ignore_index=0):
    batch_size = batched_sequence_data.size(0)
    targets = batched_sequence_data.squeeze()
    
    preds = torch.argmax(logits, dim=-1)

    correct = preds == targets
    correct[targets == ignore_index] = True
    elem_acc = correct[targets != 0].float().mean()
    sequence_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, sequence_acc

def compute_sequence_cross_entropy(logits, batched_sequence_data, data_name, string_type, is_token=False, vocab_size=200):
    logits = logits[:,:-1]
    targets = batched_sequence_data[:,1:]
    weight_vector = [0,0]
    if is_token:
        tokens = TOKENS_SPM_DICT[f'{data_name}_{string_type}_{vocab_size}']['tokens']
    elif string_type == 'adj_list':
        tokens = TOKENS_DICT[data_name]
    elif string_type == 'adj_list_diff':
        tokens = TOKENS_DICT_DIFF[data_name]
    elif string_type in ['adj_flatten', 'adj_flatten_sym', 'bwr']:
        tokens = TOKENS_DICT_FLATTEN[data_name]
    elif string_type in ['adj_seq', 'adj_seq_rel']:
        tokens = TOKENS_DICT_SEQ[data_name]    
        
    weight_vector.extend([1/(len(tokens)-2) for _ in range(len(tokens)-2)])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                        weight=torch.FloatTensor(weight_vector).to(logits.device))
    return loss

def compute_sequence_cross_entropy_feature(logits, batched_sequence_data, data_name, string_type):
    logits = logits[:,:-1]
    targets = batched_sequence_data[:,1:]
    weight_vector = [0,0]
    if string_type in ['adj_list', 'adj_list_diff']:
        tokens = TOKENS_DICT_MOL[data_name]
    elif string_type in ['adj_flatten', 'adj_flatten_sym', 'bwr']:
        tokens = TOKENS_DICT_FLATTEN_MOL[data_name]
    elif string_type in ['adj_seq', 'adj_seq_rel']:
        tokens = TOKENS_DICT_SEQ_MOL[data_name]    
        
    weight_vector.extend([1/(len(tokens)-2) for _ in range(len(tokens)-2)])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                        weight=torch.FloatTensor(weight_vector).to(logits.device))
    return loss

def process_tensor(x, y):
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x, y

def emd(x, y, distance_scaling=1.0):
    # -------- convert histogram values x and y to float, and make them equal len --------
    x = x.astype(float)
    y = y.astype(float)
    support_size = max(len(x), len(y))
    # -------- diagonal-constant matrix --------
    d_mat = toeplitz(range(support_size)).astype(float)  
    distance_mat = d_mat / distance_scaling
    x, y = process_tensor(x, y)

    emd_value = pyemd.emd(x, y, distance_mat)
    return np.abs(emd_value)

def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """ Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
      x, y: 1D pmf of two distributions with the same support
      sigma: standard deviation
    """
    emd_value = emd(x, y, distance_scaling)
    return np.exp(-emd_value * emd_value / (2 * sigma * sigma))

def gaussian(x, y, sigma=1.0):
    x = x.astype(float)
    y = y.astype(float)
    x, y = process_tensor(x, y)
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def load_eval_settings(data, orbit_on=True):
    # Settings for generic graph generation
    methods = ['degree', 'cluster', 'orbit', 'spectral'] 
    kernels = {'degree':gaussian_emd, 
                'cluster':gaussian_emd, 
                'orbit':gaussian,
                'spectral':gaussian_emd}
    return methods, kernels

def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)

def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d

def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    """ Discrepancy between 2 samples
    """
    d = 0
    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dist in executor.map(kernel_parallel_worker,
                                     [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1]):
                d += dist
                
    d /= len(samples1) * len(samples2)

    return d

def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """ MMD between two samples
    """
    # -------- normalize histograms into pmf --------
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    return disc(samples1, samples1, kernel, *args, **kwargs) + \
        disc(samples2, samples2, kernel, *args, **kwargs) - \
        2 * disc(samples1, samples2, kernel, *args, **kwargs)

def degree_worker(G):
    return np.array(nx.degree_histogram(G))

PRINT_TIME = False 

# -------- Compute degree MMD --------
def degree_stats(graph_ref_list, graph_pred_list, KERNEL=gaussian_emd, is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # -------- in case an empty graph is generated --------
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def spectral_worker(G):
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


# -------- Compute spectral MMD --------
def spectral_stats(graph_ref_list, graph_pred_list, KERNEL=gaussian_emd, is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


# -------- Compute clustering coefficients MMD --------
def clustering_stats(graph_ref_list, graph_pred_list, KERNEL=gaussian, bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    try:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL, 
                            sigma=1.0 / 10, distance_scaling=bins)
    except:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL, sigma=1.0 / 10)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist

ORCA_DIR = 'evaluation/orca'  
COUNT_START_STR = 'orbit counts: \n'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    tmp_file_path = os.path.join(ORCA_DIR, f'tmp-{random.random():.4f}.txt')
    f = open(tmp_file_path, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    output = sp.check_output([os.path.join(ORCA_DIR, 'orca'), 'node', '4', tmp_file_path, 'std'])
    output = output.decode('utf8').strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_file_path)
    except OSError:
        pass

    return node_orbit_counts

def orbit_stats_all(graph_ref_list, graph_pred_list, KERNEL=gaussian):
    total_counts_ref = []
    total_counts_pred = []

    prev = datetime.now()

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except Exception as e:
            print(e)
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except:
            print('orca failed')
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=KERNEL,
                           is_hist=False, sigma=30.0)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing orbit mmd: ', elapsed)
    return mmd_dist

### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
def compute_nspdk_mmd(samples1, samples2, metric, is_hist=True, n_jobs=None):
    def kernel_compute(X, Y=None, is_hist=True, metric='linear', n_jobs=None):
        for graph in X:
            edge_attr = nx.get_edge_attributes(graph, 'edge_attr')
            nx.set_edge_attributes(graph, edge_attr, 'label')
            node_attr = nx.get_node_attributes(graph, 'x')
            nx.set_node_attributes(graph, node_attr, 'label')
        X = vectorize(X, complexity=4, discrete=True)
        
        if Y is not None:
            Y = vectorize(Y, complexity=4, discrete=True)
        return pairwise_kernels(X, Y, metric='linear', n_jobs=n_jobs)

    X = kernel_compute(samples1, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Y = kernel_compute(samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Z = kernel_compute(samples1, Y=samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)

    return np.average(X) + np.average(Y) - 2 * np.average(Z)

##### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/stats.py
def nspdk_stats(graph_ref_list, graph_pred_list):
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    mmd_dist = compute_nspdk_mmd(graph_ref_list, graph_pred_list_remove_empty, metric='nspdk', is_hist=False, n_jobs=20)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist

METHOD_NAME_TO_FUNC = {
    'degree': degree_stats,
    'cluster': clustering_stats,
    'orbit': orbit_stats_all,
    'spectral': spectral_stats,
    'nspdk': nspdk_stats
}


# -------- Evaluate generated generic graphs --------
def eval_graph_list(graph_ref_list, graph_pred_list, methods=None, kernels=None):
    if methods is None:
        methods = ['degree', 'cluster', 'orbit']
    results = {}
    for method in methods:
        if method == 'nspdk':
            results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list)
        else:
            results[method] = round(METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list, kernels[method]), 6)
        print('\033[91m' + f'{method:9s}' + '\033[0m' + ' : ' + '\033[94m' +  f'{results[method]:.6f}' + '\033[0m')
    return results

def check_generated_samples(dataset_name='GDSS_com', string_type='adj_seq_rel', order='C-M'):
    '''
    Evaluate generated samples from the best model (json)
    '''
    with open(os.path.join("resource", f'best_model.json')) as f:
        data = json.load(f)
    ts = data[dataset_name][string_type]
    with open(os.path.join("samples", f'graphs/{dataset_name}/{ts}.pkl'), 'rb') as f:
        sampled_graphs = pickle.load(f)
    _, _, test_graphs = load_graphs(dataset_name, order)
    methods, kernels = load_eval_settings('')
    # results = eval_graph_list(test_graphs, sampled_graphs[:len(test_graphs)], methods=methods, kernels=kernels)
    results = {}
    for graph in test_graphs:
        nx.set_node_attributes(graph, 0, "label")
        nx.set_edge_attributes(graph, 1, "label")
    for graph in sampled_graphs:
        nx.set_node_attributes(graph, 0, "label")
        nx.set_edge_attributes(graph, 1, "label")

    scores_nspdk = eval_graph_list(test_graphs, sampled_graphs[:len(test_graphs)], methods=['nspdk'])['nspdk']
    results['nspdk'] = scores_nspdk
    if dataset_name == 'planar':
        results['validity'] = eval_acc_planar_graph(sampled_graphs)
    elif dataset_name == 'GDSS_grid':
        results['validity'] = eval_acc_grid_graph(sampled_graphs)
    elif dataset_name == 'sbm':
        results['validity'] = eval_acc_sbm_graph(sampled_graphs)
    print(results)