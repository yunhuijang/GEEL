import pickle

data_name = ''
with open(f'resource/{data_name}.pkl', 'rb') as f:
    graphs = pickle.load(f)
    
print(max([len(graph.edges) for graph in graphs]))