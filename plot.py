import math
import networkx as nx
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)
matplotlib.use('Agg')



options = {
    'node_size': 5,
    'edge_color' : 'black',
    'linewidths': 0.8,
    'width': 1.2
}

options_one = {
    'node_size': 15,
    'edge_color' : 'black',
    'linewidths': 1,
    'width': 1.2
}

def plot_graphs_list(graphs, title='title', max_num=16, save_dir=None, N=0, format='png'):
    batch_size = len(graphs)
    max_num = min(batch_size, max_num)
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure()

    for i in range(max_num):
        # idx = i * (batch_size // max_num)
        idx = i + max_num*N
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)

        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f'e={e - l}, n={v}'
        # if 'lobster' in save_dir.split('/')[0]:
        #     if is_lobster_graph(graphs[idx]):
        #         title_str += f' [L]'
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
    # figure.suptitle(title)
    save_fig(save_dir=save_dir, title=title, dpi=1000, format=format)
    
    return 

def plot_one_graph(graph, title, save_dir):
    figure = plt.figure()

    G = graph
    G.remove_nodes_from(list(nx.isolates(G)))
    e = G.number_of_edges()
    v = G.number_of_nodes()
    l = nx.number_of_selfloops(G)

    fig, ax = plt.subplots(1,1, figsize=(3,3))
    plt.figure(figsize=(5,5))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, **options_one)
    # ax.title.set_text(title_str)
    # figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)

def save_fig(save_dir=None, title='title', dpi=1000, format='png'):
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_dir is None:
        plt.show()
    else:
        fig_dir = os.path.join(*['samples', 'fig', save_dir])
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, f'{title}.{format}'),
                    bbox_inches='tight',
                    dpi=dpi,
                    transparent=False,
                    format=format)
        plt.close()
    return


def save_graph_list(log_folder_name, exp_name, gen_graph_list):

    if not(os.path.isdir('./samples/pkl/{}'.format(log_folder_name))):
        os.makedirs(os.path.join('./samples/pkl/{}'.format(log_folder_name)))
    with open('./samples/pkl/{}/{}.pkl'.format(log_folder_name, exp_name), 'wb') as f:
            pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    save_dir = './samples/pkl/{}/{}.pkl'.format(log_folder_name, exp_name)
    return save_dir
