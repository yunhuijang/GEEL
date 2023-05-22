from plot import plot_graphs_list, plot_one_graph, save_fig
import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

from data.mol_utils import smiles_to_mols
from rdkit.Chem import Draw


data_name = 'GDSS_com'
method = 'digress'

gcg_dict = {'GDSS_com': 'May02-00:49:13', 'GDSS_grid': 'May02-15:56:32', 'GDSS_ego': "May14-07:56:26",
            'GDSS_enz': 'May02-06:39:38' , 'planar': "May14-11:34:11"}

graphgen_dict = {'GDSS_com': 'DFScodeRNN_com_small_2023-05-14 01:40:56', 
                 'GDSS_grid': 'DFScodeRNN_grid_2023-05-14 01:43:04', 
                 'GDSS_ego': "DFScodeRNN_ego_small_2023-05-14 01:39:50",
                 'GDSS_enz': 'DFScodeRNN_enz_2023-05-14 02:11:21',
                 'planar': "DFScodeRNN_planar_2023-05-14 01:53:01"}

digress_dict = {'GDSS_com': '2023-05-13/14-54-28', 
                 'GDSS_grid': '2023-05-13/15-02-58', 
                 'GDSS_ego': '2023-05-13/14-53-38',
                 'GDSS_enz': '2023-05-13/14-56-45',
                 'planar': "2023-05-13/13-49-30"}

def draw_generated_graphs(data_name, method, index=0):
    if method == 'train':
        # for train
        with open(f'resource/{data_name}/C-M/{data_name}_test_graphs.pkl', 'rb') as f:
            graphs = pickle.load(f)
            
    elif method == 'digress':
        # digress
        file_name = digress_dict[data_name]
        with open(f'../DiGress/src/outputs/{file_name}/graphs/generated_graphs.pkl', 'rb') as f:
            graphs = pickle.load(f)
        adjs = [graph[1] for graph in graphs]
        graphs = [nx.from_numpy_array(adj.numpy()) for adj in adjs]
        
    elif method == 'graphgen':
        file_name = graphgen_dict[data_name]
        with open(f'../graphgen/graphs/{file_name}/generated_graphs.pkl', 'rb') as f:
            graphs = pickle.load(f)
            
    elif method == 'gcg':
        file_name = gcg_dict[data_name]
        with open(f'samples/graphs/{data_name}/{file_name}.pkl', 'rb') as f:
            graphs = pickle.load(f)
            
    elif method == 'gdss':
        data_dict = {'GDSS_com': 'community_small', 'GDSS_enz': 'ENZYMES', 'GDSS_grid': 'grid'}
        with open(f'../GDSS/samples/pkl/{data_dict[data_name]}/test/{data_name}_sample.pkl', 'rb') as f:
            graphs = pickle.load(f)
        print(len(graphs))
    plot_graphs_list(graphs[:9], title=f'{data_name}-{method}', save_dir=f'figure/{data_name}', max_num=9)
    # plot_one_graph(graphs[index], title=f'{method}-one', save_dir=f'figure/{data_name}')

def draw_generated_molecules(data_name):
    if data_name == 'qm9':
        with open("samples/smiles/qm9/May09-07:00:25.txt", 'r') as f:
            smiles = f.readlines()
    elif data_name == 'zinc':
        with open("samples/smiles/zinc/May06-13:30:46.txt", 'r') as f:
            smiles = f.readlines()
    
    mols = smiles_to_mols(smiles[:24])
    img = Draw.MolsToGridImage(mols, molsPerRow=8)
    img.save(f"samples/fig/figure/{data_name}/{data_name}.png")
    

def draw_loss_plot():
    df = pd.read_csv('resource/planar_ab_pe.csv')
    fig, ax = plt.subplots()
    x = np.arange(0,500,1/5)
    x_y_spline_tpe = make_interp_spline(x, df['tpe'].dropna())
    x_tpe = np.linspace(x.min(), x.max(), 100)
    y_tpe = x_y_spline_tpe(x_tpe)

    x_y_spline_ape = make_interp_spline(x, df['ape'].dropna())
    x_ape = np.linspace(x.min(), x.max(), 100)
    y_ape = x_y_spline_ape(x_ape)

    x_y_spline_rpe = make_interp_spline(x, df['rpe'].dropna())
    x_rpe = np.linspace(x.min(), x.max(), 100)
    y_rpe = x_y_spline_rpe(x_rpe)

    ax.plot(x_tpe, y_tpe, label='TPE', color='#F8C159', linewidth=3)
    ax.plot(x_ape, y_ape, label='APE', color='#4384C2', linewidth=3)
    ax.plot(x_rpe, y_rpe, label='RPE', color='#EF4C56', linewidth=3)
    ax.legend(fontsize=13)
    ax.set_xlabel('Epochs', fontsize=13)
    ax.set_ylabel('Training loss', fontsize=13)
    ax.grid(linestyle='dotted')
    for pos in ['left', 'right', 'top', 'bottom']:
        ax.spines[pos].set_linewidth(1.5)
    
# for data in ['GDSS_com', 'GDSS_enz', 'GDSS_grid']:
# # for data in ['GDSS_grid']:
#     print(data)
#     # for method in ['train', 'gcg', 'digress', 'graphgen']:
#     for method in ['gdss']:
#         print(method)
#         draw_generated_graphs(data, method, 0)

draw_generated_molecules('zinc')