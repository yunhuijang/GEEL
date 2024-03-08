import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
from tqdm import tqdm

from data.tokens import map_tokens, token_to_id, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, id_to_token


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class SimpleTokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(SimpleTokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, pe, max_len, string_type, data_name, bw, num_nodes, order):
        super(TokenEmbedding, self).__init__()

        self.num_nodes = num_nodes
        self.bw = bw
        self.emb_size = emb_size
        self.pe = pe
        self.data_name = data_name
        self.vocab_size = vocab_size
        self.order = order
        
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embedding_numnode = nn.Embedding(self.num_nodes+3, emb_size)
        self.embedding_diff = nn.Embedding(self.bw+4, emb_size)
        
        # max_len+2: for eos / bos token
        self.positional_embedding = nn.Parameter(torch.randn([1, max_len+2, emb_size]))

        self.string_type = string_type
        self.tokens = map_tokens(self.data_name, self.string_type, self.vocab_size-3, self.order)
        self.ID2TOKEN = id_to_token(self.tokens)
        self.data_name = data_name
    
    def split_nodes(self, id_to_token, token_sequences, device):
        mapping_tensor1 = torch.zeros(len(id_to_token), dtype=torch.long)
        mapping_tensor2 = torch.zeros(len(id_to_token), dtype=torch.long)

        for key, value in id_to_token.items():
            if isinstance(value, tuple):
                mapping_tensor1[key] = value[0] + 3
                mapping_tensor2[key] = value[1] + 3
            else:
                # pad, bos, eos
                mapping_tensor1[key] = key
                mapping_tensor2[key] = key

        output_tokens1 = mapping_tensor1[token_sequences]
        output_tokens2 = mapping_tensor2[token_sequences]
    
        return output_tokens1.to(device), output_tokens2.to(device)
        
    def forward(self, token_sequences):
        ID2TOKEN = id_to_token(self.tokens)
        if self.string_type in ['adj_flatten', 'adj_flatten_sym', 'bwr', 'adj_seq', 'adj_seq_rel', 'adj_seq_merge', 'adj_seq_rel_merge', 'adj_seq_blank', 'adj_seq_rel_blank', 'adj_list_diff_ni', 'adj_list_diff_ni_rel']:
            x = self.embedding(token_sequences) * math.sqrt(self.emb_size)
        elif self.string_type in ['adj_list_diff', 'adj_list']:
            t1, t2 = self.split_nodes(ID2TOKEN, token_sequences, device=token_sequences.device)
            x1 = self.embedding_numnode(t1) * math.sqrt(self.emb_size)
            if self.string_type == 'adj_list':
                x2 = self.embedding_numnode(t2) * math.sqrt(self.emb_size)
            elif self.string_type == 'adj_list_diff':
                x2 = self.embedding_diff(t2) * math.sqrt(self.emb_size)
            x = x1 + x2
            
        # node PE for adj_list_diff_ni (cumsum)
        if self.pe == 'node':
            ni, _ = self.split_nodes(ID2TOKEN, token_sequences, device=token_sequences.device)
            ni[ni==2] = -100000
            zero_mask = torch.isin(ni, torch.tensor([0,1,3]).to(token_sequences.device))
            ni.masked_fill_(zero_mask, 0)
            ni[ni>0] -= 3
            current_node = ni.cumsum(dim=1)
            current_node[current_node<0] = 0
            current_node[current_node > self.num_nodes+1] = 0
            node_pe = self.embedding_numnode(current_node) * math.sqrt(self.emb_size)
            x += node_pe
        # learnable PE
        elif self.pe == 'learn':
            x_batch_size = x.shape[0]
            x_seq_len = x.shape[1]
            pe = self.positional_embedding[:,:x_seq_len]
            pe_stack = torch.tile(pe, (x_batch_size, 1, 1))
            return x + pe_stack
        
        elif self.pe == 'no':
            pass
        
        elif self.pe == 'rel':
            pass
        
        return x

class LSTMGenerator(nn.Module):
    def __init__(
        self, emb_size, dropout, num_layers, string_type, dataset, vocab_size, num_nodes, max_len, bw, is_token=False, pe='node', is_simple_token=False, order='C-M'
    ):
        super(LSTMGenerator, self).__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.string_type = string_type
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.is_token = is_token
        self.pe = pe
        self.max_len = max_len
        self.bw = bw
        self.num_nodes = num_nodes
        self.is_simple_token = is_simple_token
        self.order = order
        
        self.tokens = map_tokens(self.dataset, self.string_type, self.vocab_size, self.order, self.is_token)

        self.TOKEN2ID = token_to_id(self.dataset, self.string_type, self.is_token, self.vocab_size, self.order)
        self.vocab_size = self.output_size = len(self.tokens)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.emb_size)
        self.token_embedding_layer = TokenEmbedding(self.vocab_size, emb_size, self.pe, self.max_len, self.string_type, self.dataset, self.bw, self.num_nodes, self.order)
        self.simple_token_embedding_layer = SimpleTokenEmbedding(self.vocab_size, self.emb_size)
        self.lstm_layer = nn.LSTM(self.emb_size, self.emb_size,  
                                dropout=self.dropout, batch_first=True, num_layers=self.num_layers)
        self.linear_layer = nn.Linear(self.emb_size, self.output_size)


    def forward(self, sequences):
        batch_size = sequences.size(0)
        if self.is_simple_token:
            out = self.simple_token_embedding_layer(sequences)    
        else:
            out = self.token_embedding_layer(sequences)
        out, hidden = self.lstm_layer(out)
        logits = self.linear_layer(out)

        return logits

    def decode(self, num_samples, max_len, device):
        '''
        sequential generation
        '''
        
        # first element need to be (1,X)
        first_indices = [value for key, value in self.TOKEN2ID.items() if (type(key) is tuple) and (key[0] == 1)]
        
        sequences = torch.LongTensor([[self.TOKEN2ID[BOS_TOKEN]] for _ in range(num_samples)]).to(device)
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        for index in tqdm(range(max_len), "generation"):
            if ended.all():
                break
            logits = self(sequences)
            target_logits = logits[:, -1]
            
            if self.string_type in ['adj_list_diff_ni', 'adj_list_diff_ni_rel']:
                # force the first token to be (1, X)
                if index == 0:
                    current_logits = logits[:, -1].clone().detach()
                    target_logits = torch.full(logits[:, -1].shape, float("-inf"), device=logits.device)
                    target_logits[:, first_indices] = current_logits[:, first_indices]
            
            preds = Categorical(logits=target_logits).sample()
            preds[ended] = self.TOKEN2ID[PAD_TOKEN]
            sequences = torch.cat([sequences, preds.unsqueeze(1)], dim=1)

            ended = torch.logical_or(ended, preds == self.TOKEN2ID[EOS_TOKEN])
        return sequences