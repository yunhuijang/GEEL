import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
from tqdm import tqdm
import math
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from data.tokens import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, map_tokens
from data.mol_tokens import token_to_id_mol,  NODE_TOKENS_DICT, NODE_TYPE_DICT, BOND_TYPE_DICT


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, data_name, string_type, num_nodes, order):
        super(TokenEmbedding, self).__init__()
        self.emb_size = emb_size
        self.data_name = data_name
        self.string_type = string_type
        self.num_nodes = num_nodes
        self.order = order
        
        self.tokens = map_tokens(self.data_name, self.string_type, None, self.order)
        TOKEN2ID = token_to_id_mol(self.data_name, self.string_type)
        self.edge_indices = [value for key, value in TOKEN2ID.items() if (type(key) is tuple)]
        self.edge_indices_ni = [value for key, value in TOKEN2ID.items() if (type(key) is tuple) and (key[0] == 1)]
        
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embedding_numnode = nn.Embedding(self.num_nodes+3, emb_size)
        
    def forward(self, token_sequences):
        x = self.embedding(token_sequences.long()) * math.sqrt(self.emb_size)
        # node PE for adj_list_diff_ni
        if self.string_type == 'adj_list_diff_ni':
            ni = token_sequences.clone().detach()
            ni_mask = torch.isin(ni, torch.tensor(self.edge_indices_ni).to(token_sequences.device))
            edge_mask = torch.isin(ni, torch.tensor(self.edge_indices).to(token_sequences.device))
            eos_mask = torch.isin(ni, torch.tensor([2]).to(token_sequences.device))
            ni.masked_fill_(ni_mask, 1)
            ni.masked_fill_(~ni_mask, 0)
            ni.masked_fill_(eos_mask, -100000)
            current_node = ni.cumsum(dim=1)
            current_node[current_node<0] = 0
            current_node[current_node > self.num_nodes+1] = 0
            current_node.masked_fill_(~edge_mask, 0)
            node_pe = self.embedding_numnode(current_node) * math.sqrt(self.emb_size)
            x += node_pe
            
        return x
# model with that generates feature + adj list (masking in sequence)
class LSTMGeneratorFeatureList(nn.Module):
    def __init__(
        self, emb_size, dropout, num_layers, string_type, dataset, vocab_size, num_nodes, max_len, bw, is_token=False, learn_pos=False, is_simple_token=False, order='C-M'
    ):
        super(LSTMGeneratorFeatureList, self).__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.string_type = string_type
        self.data_name = dataset
        self.vocab_size = vocab_size
        self.is_token = is_token
        self.learn_pos = learn_pos
        self.max_len = max_len
        self.bw = bw
        self.num_nodes = num_nodes
        self.is_simple_token = is_simple_token
        self.order = order
        
        self.tokens = map_tokens(self.data_name, self.string_type, self.vocab_size, self.order, self.is_token)

        self.vocab_size = self.output_size = len(self.tokens)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.emb_size)
        self.token_embedding_layer = TokenEmbedding(self.vocab_size, emb_size, self.data_name, self.string_type, self.num_nodes, self.order)
        self.lstm_layer = nn.LSTM(self.emb_size, self.emb_size,  
                                dropout=self.dropout, batch_first=True, num_layers=self.num_layers)
        self.linear_layer = nn.Linear(self.emb_size, self.output_size)


    def forward(self, sequences):
        batch_size = sequences.size(0)
        out = self.token_embedding_layer(sequences)
        out, hidden = self.lstm_layer(out)
        logits = self.linear_layer(out)

        return logits

    def decode(self, num_samples, max_len, device):
        '''
        sequential generation
        '''
        TOKEN2ID = token_to_id_mol(self.data_name, self.string_type)
        node_type_list = NODE_TOKENS_DICT[self.data_name]
        node_type_indices = [TOKEN2ID[node_type] for node_type in node_type_list]
        # node_types = [NODE_TYPE_DICT[node_type] for node_type in node_type_list]
        # node_type_indices = [TOKEN2ID[node_type] for node_type in node_types]
        edge_types = BOND_TYPE_DICT.values()
        edge_type_indices = [TOKEN2ID[bond_type] for bond_type in edge_types]
        edge_indices = [value for key, value in TOKEN2ID.items() if type(key) is tuple]
        # edge starts with 1
        edge_indices_ni = [value for key, value in TOKEN2ID.items() if (type(key) is tuple) and (key[0] == 1)]
        # edge starts with 0
        edge_indices_non_ni = [value for key, value in TOKEN2ID.items() if (type(key) is tuple) and (key[0] == 0)]
        sequences = torch.LongTensor([[TOKEN2ID[BOS_TOKEN]] for _ in range(num_samples)]).to(device)
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        for index in tqdm(range(max_len), "generation"):
            if ended.all():
                break
            current_tokens = sequences[:, index]
            is_node_type = sum(current_tokens==i for i in node_type_indices).bool()
            is_edge_type = sum(current_tokens==i for i in edge_type_indices).bool()
            is_edge = sum(current_tokens==i for i in edge_indices).bool()
                
            logits = self(sequences)
            
            current_logits = logits[:, -1].clone().detach()
            
            # leave only node types
            node_type_logits = torch.full(logits[:, -1].shape, float("-inf"), device=logits.device)
            node_type_logits[:, node_type_indices] = current_logits[:, node_type_indices]
            
            
            # leave only edge types
            edge_type_logits = torch.full(logits[:, -1].shape, float("-inf"), device=logits.device)
            edge_type_logits[:, edge_type_indices] = current_logits[:, edge_type_indices]
            
            # leave only edges
            edge_logits = torch.full(logits[:, -1].shape, float("-inf"), device=logits.device)
            edge_logits[:, edge_indices] = current_logits[:, edge_indices]
            
            # node type + edge: after edge type
            node_type_edge_logits = node_type_logits.clone()
            node_type_edge_logits[:, edge_indices] = current_logits[:, edge_indices]
            node_type_edge_logits[:, TOKEN2ID[EOS_TOKEN]] = current_logits[:, TOKEN2ID[EOS_TOKEN]]
            
            # leave only edges that starts with 1
            edge_ni_logits = torch.full(logits[:, -1].shape, float("-inf"), device=logits.device)
            edge_ni_logits[:, edge_indices_ni] = current_logits[:, edge_indices_ni]
            
            # node type + edge that starts with 0: after edge type for diff NI
            node_type_edge_nonni_logits = node_type_logits.clone()
            node_type_edge_nonni_logits[:, edge_indices_non_ni] = current_logits[:, edge_indices_non_ni]
            node_type_edge_nonni_logits[:, TOKEN2ID[EOS_TOKEN]] = current_logits[:, TOKEN2ID[EOS_TOKEN]]
            
            if index == 0:
                target_logits = node_type_logits
            elif index == 1:
                if self.string_type == 'adj_list_diff_ni':
                    target_logits = edge_ni_logits
                else:
                    target_logits = node_type_logits
            else:
                if self.string_type == 'adj_list_diff_ni':
                    target_logits = torch.where(is_node_type, edge_ni_logits.T, node_type_logits.T).T
                    target_logits = torch.where(is_edge, edge_type_logits.T, target_logits.T).T
                    target_logits = torch.where(is_edge_type, node_type_edge_nonni_logits.T, target_logits.T).T
                else:
                    target_logits = torch.where(is_node_type, edge_logits.T, node_type_logits.T).T
                    target_logits = torch.where(is_edge, edge_type_logits.T, target_logits.T).T
                    target_logits = torch.where(is_edge_type, node_type_edge_logits.T, target_logits.T).T
                    
            preds = Categorical(logits=target_logits).sample()
            preds[ended] = TOKEN2ID[PAD_TOKEN]
            sequences = torch.cat([sequences, preds.unsqueeze(1)], dim=1)

            ended = torch.logical_or(ended, preds == TOKEN2ID[EOS_TOKEN])

        return sequences