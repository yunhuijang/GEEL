from data.data_utils import load_graphs, create_graphs, get_max_len
import networkx as nx
from data.orderings import bw_from_adj

n = [32, 71, 100, 224, 316]
num = [1000, 5000, 10000, 50000, 100000]

# for size, nod_num in zip(n, num):
    # graphs = create_graphs(f'grid{size}', nod_num)

