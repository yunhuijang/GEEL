import pickle
from data.data_utils import get_max_len, load_graphs

# for data in ['GDSS_ego', 'GDSS_com', 'GDSS_enz', 'GDSS_grid', 'planar', 'sbm', 'proteins']:
# # for data in ['planar', 'sbm']:
#     print(data)
#     print(get_max_len(data))

graphs = load_graphs('proteins')
for graph in graphs:
    print(len(graph))

# for i, graph in enumerate(graphs[0]):
#     if 185 in set(graph.nodes):
#         print(len(graph.nodes))

# print(get_max_len('sbm'))