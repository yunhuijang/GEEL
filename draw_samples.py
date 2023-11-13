from plot import plot_graphs_list, plot_one_graph, save_fig
import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from math import log
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import Divider, Size



from data.mol_utils import smiles_to_mols
from rdkit.Chem import Draw


data_name = 'GDSS_com'
method = 'digress'

gcg_dict = {'GDSS_com': 'May02-00:49:13', 'GDSS_grid': 'May02-15:56:32', 'GDSS_ego': "May14-07:56:26",
            'GDSS_enz': 'May02-06:39:38' , 'planar': "May14-11:34:11"}

graphgen_dict = {'GDSS_com': 'DFScodeRNN_com_small_2023-05-14 01:40:56', 
                 'GDSS_grid': 'DFScodeRNN_grid_2023-05-14 01:43:04', 
                 'GDSS_ego': "DFScodeRNN_ego_small_2023-05-14 01:39:400",
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
        with open("samples/smiles/qm9/iupdgobu.txt", 'r') as f:
            smiles = f.readlines()[:24]
    elif data_name == 'zinc':
        with open("samples/smiles/zinc/May06-13:30:46.txt", 'r') as f:
            smiles = f.readlines()
    
    mols = smiles_to_mols(smiles[:24])
    img = Draw.MolsToGridImage(mols, molsPerRow=8)
    img.save(f"samples/fig/{data_name}/{data_name}.pdf")
    

def draw_loss_plot():
    df = pd.read_csv('resource/ordering_loss.csv')
    # df = df.applymap(lambda x: log(x))
    fig, ax = plt.subplots(figsize=(5,4))
    
    x = np.arange(0,2000, 1/2)
    x_y_spline_tpe = make_interp_spline(x, df['Random'].dropna()[:4000])
    x_tpe = np.linspace(x.min(), x.max(), 400)
    y_tpe = x_y_spline_tpe(x_tpe)

    # x = np.arange(0,len(df['BFS'].dropna()))
    x_y_spline_ape = make_interp_spline(x, df['BFS'].dropna()[:4000])
    x_ape = np.linspace(x.min(), x.max(), 400)
    y_ape = x_y_spline_ape(x_ape)

    # x = np.arange(0,len(df['DFS'].dropna()))
    x_y_spline_rpe = make_interp_spline(x, df['DFS'].dropna()[:4000])
    x_rpe = np.linspace(x.min(), x.max(), 400)
    y_rpe = x_y_spline_rpe(x_rpe)
    
    # x = np.arange(0,len(df['C-M'].dropna()))
    x_y_spline_cm = make_interp_spline(x, df['C-M'].dropna()[:4000])
    x_cm = np.linspace(x.min(), x.max(), 400)
    y_cm = x_y_spline_cm(x_cm)


    ax.plot(x_cm, y_cm, label='C-M ($B$=19)', color='#41BC9E', linewidth=3)
    ax.plot(x_ape, y_ape, label='BFS ($B$=33)', color='#4384C2', linewidth=3)
    ax.plot(x_tpe, y_tpe, label='Random ($B$=347)', color='#F8C159', linewidth=3)
    ax.plot(x_rpe, y_rpe, label='DFS ($B$=360)', color='#EF4C56', linewidth=3)
    
    ax.legend(fontsize=14, loc='upper right')
    ax.set_xlabel('Epochs', fontsize=18)
    ax.set_ylabel('Training loss', fontsize=18)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    ax.grid(linestyle='dotted')
    for pos in ['left', 'right', 'top', 'bottom']:
        ax.spines[pos].set_linewidth(1.5)
    
    format = 'pdf'
    save_fig('fig/', f'ordering_loss.{format}', dpi=1000, format=format)
    
def draw_time_theory_plot():
    df = pd.read_csv('resource/gen_time.csv', header=0)
    fig, ax = plt.subplots(figsize=(7,4))
    
    x = df['node size']
    
    y = df['time (s)']
    
    ax.plot(x, y, marker='o', color='#EF4C56', label='GEEL', linewidth=3)
    ax.plot(x, 2*x, color='#41BC9E', linestyle='--', label='$c_2*M$', linewidth=3)
    ax.plot(x, 0.5*x, color='#4384C2', linestyle='--', label='$c_1*M$', linewidth=3)
    
    ax.legend(fontsize=14, loc='upper left')
    
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(labels=[1, 10, 100, 1000, 10000], fontsize=15)
    ax.set_xscale('log')
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(labels=[1, 10, 100, 1000, 10000], fontsize=15)
    ax.set_yscale('log')
    
    ax.set_xlabel('Number of nodes', fontsize=18)
    ax.set_ylabel('Generation time (sec)', fontsize=18)
    ax.grid(linestyle='dotted')
    
    plt.minorticks_off()
    
    form = 'pdf'
    save_fig('fig/', f'time_theory.{form}', dpi=1000, format=form)

def draw_time_dot_plot():
    df = pd.read_csv('resource/gen_real_time.csv', header=0)
    fig, ax = plt.subplots(figsize=(5,2.5))
    x = np.arange(0, 3/2, 1/2)
    print(df.columns)
    colors = ['#4384C2', '#41BC9E', '#7030A0', '#EF4C56', '#F8C159']

    for model, color in zip(df.columns[1:], colors):
        if model == 'GEEL+LSTM':
            ax.plot(x, df[model]/10, marker='o', label='GEEL', linestyle="-", linewidth=4, markersize=10, color=color)
        else:
            ax.plot(x, df[model]/10, marker='o', label=model, linestyle="-", linewidth=4, markersize=10, color=color)
    # ax.plot(x, df['GEEL+LSTM'], marker='o', label='GEEL+LSTM', linestyle="", markersize=10)
    # else:
    # ax.plot(x, df['GRAN'], marker='*', label='GRAN', linestyle="", markersize=10, color='gray')
    # ax.plot(x, df['GDSS'], marker='x', label='GDSS', linestyle="", markersize=10, color='gray')
    # ax.plot(x, df['BiGG'], marker='D', label='GDSS', linestyle="", markersize=10, color='gray')
    # ax.plot(x, df['DiGress'], marker='v', label='DiGress', linestyle="", markersize=10, color='gray')
    
    ax.set_yticks([0.1, 1, 10, 100, 1000])
    ax.set_yticklabels(labels=[0.1, 1, 10, 100, 1000], fontsize=14)
    ax.set_yscale('log')
    # ax.set_xticks(['Planar', 'Enzymes', 'Grid'])
    ax.set_xticks(x, ['Enzymes', 'Planar', 'Grid'])
    ax.set_xticklabels(labels=['Enzymes', 'Planar', 'Grid'], fontsize=15)
    ax.set_ylabel('Generation time (s)', fontsize=13)
    
    
    formatter = ticker.FuncFormatter(lambda y, _: '{:.1f}'.format(y)) 
    # ax.ticklabel_format(style='plain', axis='x')
    ax.yaxis.set_major_formatter(formatter)
    
    # ax.set_xlabel('Dataset', fontsize=18)
    ax.grid(linestyle='dotted')
    ax.legend(fontsize=10, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.35))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1/2)) 
    plt.minorticks_off()
    form = 'pdf'
    save_fig('fig/', f'time_real_2.{form}', dpi=1000, format=form)

def draw_size_plot():
    df = pd.read_csv('resource/rep_vocab_size_2.csv', header=0)
    fig, ax = plt.subplots(figsize=(5,2.5))
    x = np.arange(0, 1, 1/3)
    width = 0.1 
    multiplier = 0
    # h = [Size.Fixed(1.0), Size.Fixed(5.5)]
    # v = [Size.Fixed(0.7), Size.Fixed(2.)]

    # divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    # # The width and height of the rectangle are ignored.

    # ax = fig.add_axes(divider.get_position(),
    #                 axes_locator=divider.new_locator(nx=1, ny=1))
    
    ax.set_yticks([1, 10, 100])
    ax.set_yscale('log')
    colors = ['#4384C2', '#F8C159']
    columns = ['rep_size', 'vocab_size']
    labels = ['Repr. '+r"($\frac{M}{N^2}$)", 'Vocab. '+r"($\frac{B^2}{N^2}$)"]
    # labels = [r"$\frac{1}{2}$", r"$\frac{1}{2}$"]
    for col, label, color in zip(columns, labels, colors):
        offset = width * multiplier
        rects = ax.bar(x+offset, 100*round(df[col]/df['N^2'],3), label=label, width=width, color=color)
        # ax.bar_label(rects, fmt="%.1f%%", padding=-20)
        multiplier += 1
    
    # ax.set_yticklabels(labels=[1, 10, 100])
    ax.set_xticks(x+0.5*width, ['Enzymes', 'Planar', 'Grid'])
    ax.set_xticklabels(labels=['Enzymes', 'Planar', 'Grid'], fontsize=13)
    # ax.set_yticklabels
    ax.set_ylabel('Reduction rate (%)', fontsize=13)
    plt.yticks(fontsize=14)
    ax.legend(fontsize=11, ncol=1, loc='upper right')
    ax = plt.gca()
    ax.set_ylim([0, 100])
    # for model in df.columns[1:]:
        # ax.plot(x, df[model], marker='o', label=model, linestyle="", markersize=10)
    # ax.bar(x, df['N^2'], label='$N^2$')
    # ax.bar(x, df['rep_size'], label='Rep. size')
    # ax.bar(x, df['vocab_size'], label='Vocab. size')
    plt.minorticks_off()
    ax.set_yticklabels(labels=[0, 0, 1, 10, 100])
    format = 'pdf'
    save_fig('fig/', f'rep_vocab_size.{format}', dpi=1000, format=format)

def draw_num_node_plot():
    df = pd.read_csv('resource/num_node.csv', header=0)
    fig, ax = plt.subplots(figsize=(5,4))
    x = np.arange(0, 4/2, 1/2)
    print(df.columns)
    colors = ['#4384C2', '#41BC9E', '#F8C159']

    for model, color in zip(df.columns[1:], colors):
        ax.plot(x, df[model]+0.00000001, marker='o', label=model, linestyle="-", linewidth=4, markersize=10, color=color)
    ax.set_yticks([0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
    ax.set_yscale('log')
    # ax.set_xticks(['Planar', 'Enzymes', 'Grid'])
    ax.set_xticks(x, ['$0.5k$', '$1k$', '$5k$', '$10k$'])
    ax.set_xticklabels(labels=['$0.5K$', '$1K$', '$5K$', '$10K$'], fontsize=18)
    ax.set_ylabel('Orbit MMD', fontsize=18)
    ax.set_xlabel('# of nodes', fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    # ax.set_xlabel('Dataset', fontsize=18)
    ax.grid(linestyle='dotted')
    ax.legend(fontsize=14, loc='lower left')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1/2)) 
    plt.minorticks_off()
    form = 'pdf'
    save_fig('fig/', f'num_node.{form}', dpi=1000, format=form)

