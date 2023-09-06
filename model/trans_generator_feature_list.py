import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
from tqdm import tqdm
import math
from time import time
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from data.tokens import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, TOKENS_DICT, TOKENS_DICT_DIFF, TOKENS_DICT_FLATTEN, TOKENS_DICT_SEQ, token_to_id, id_to_token, TOKENS_SPM_DICT, map_tokens
from data.mol_tokens import TOKENS_KEY_DICT_SEQ_MERGE_MOL, token_to_id_mol, id_to_token_mol, tokenize_mol, TOKENS_DICT_SEQ_MERGE_MOL, NODE_TOKENS_DICT
from data.data_utils import NODE_TYPE_DICT, BOND_TYPE_DICT

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    # TODO: token embedding eimension
    def __init__(self, vocab_size, emb_size, learn_pos, max_len, string_type, data_name, bw, num_nodes):
        super(TokenEmbedding, self).__init__()
        # self.num_nodes = int((-1+ math.sqrt(1 + 4*2*(vocab_size - 3))) / 2)
        self.num_nodes = num_nodes
        self.bw = bw
        self.emb_size = emb_size
        self.learn_pos = learn_pos
        self.data_name = data_name
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embedding_numnode = nn.Embedding(self.num_nodes+4, emb_size)
        self.embedding_diff = nn.Embedding(self.bw+4, emb_size)
        
        
        # max_len+2: for eos / bos token
        self.positional_embedding = nn.Parameter(torch.randn([1, max_len+2, emb_size]))

        self.string_type = string_type
        self.tokens = map_tokens(self.data_name, self.string_type, self.vocab_size-3)
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
        # TODO: if necessary, fix token embedding for adj_list (same node -> same embedding)
        # if self.string_type in ['adj_flatten', 'adj_flatten_sym', 'bwr', 'adj_seq', 'adj_seq_rel', 'adj_seq_merge', 'adj_seq_rel_merge', 'adj_seq_blank', 'adj_seq_rel_blank']:
        x = self.embedding(token_sequences) * math.sqrt(self.emb_size)
        # elif self.string_type in ['adj_list_diff', 'adj_list']:
        #     ID2TOKEN = id_to_token(self.tokens)
        #     t1, t2 = self.split_nodes(ID2TOKEN, token_sequences, device=token_sequences.device)
        #     # m = max(max(torch.flatten(t1)).item(), max(torch.flatten(t2)).item())
        #     x1 = self.embedding_numnode(t1) * math.sqrt(self.emb_size)
        #     if self.string_type == 'adj_list':
        #         x2 = self.embedding_numnode(t2) * math.sqrt(self.emb_size)
        #     elif self.string_type == 'adj_list_diff':
        #         x2 = self.embedding_diff(t2) * math.sqrt(self.emb_size)
        #     x = x1 + x2
        # learnable PE
        if self.learn_pos:
            x_batch_size = x.shape[0]
            x_seq_len = x.shape[1]
            pe = self.positional_embedding[:,:x_seq_len]
            pe_stack = torch.tile(pe, (x_batch_size, 1, 1))
            return x + pe_stack
        return x

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: shape [batch_size, seq_len, emb_size (pad_vocab_size)]
        x = x + self.pe[:x.size(1), :].transpose(0,1)
        return x
    
# model with that generates feature + adj list (masking in sequence)
class TransGeneratorFeatureList(nn.Module):
    '''
    without tree information (only string)
    '''
    
    def __init__(
        self, num_layers, emb_size, nhead, dim_feedforward, 
        input_dropout, dropout, max_len, string_type, learn_pos, abs_pos, 
        data_name, bw, num_nodes, is_token, vocab_size
    ):
        super(TransGeneratorFeatureList, self).__init__()
        self.nhead = nhead
        self.data_name = data_name
        self.string_type = string_type
        self.is_token = is_token
        self.vocab_size = vocab_size
        self.tokens = map_tokens(self.data_name, self.string_type, self.vocab_size, self.is_token)
        self.ID2TOKEN = id_to_token(self.tokens)
        
        self.learn_pos = learn_pos
        self.abs_pos = abs_pos
        self.max_len = max_len
        self.bw = bw
        self.num_nodes = num_nodes
        
        if self.abs_pos:
            self.positional_encoding = AbsolutePositionalEncoding(emb_size)
        
        self.token_embedding_layer = TokenEmbedding(len(self.tokens), emb_size, self.learn_pos, self.max_len, self.string_type, self.data_name, self.bw, self.num_nodes)
        self.input_dropout = nn.Dropout(input_dropout)
        
        #
        self.distance_embedding_layer = nn.Embedding(max_len + 1, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #
        self.generator = nn.Linear(emb_size, len(self.tokens))
        

    def forward(self, sequences):
        # for training of qm9 / zinc
        if isinstance(sequences, tuple):
            sequences = sequences[0]
        
        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)
        TOKEN2ID = token_to_id_mol(self.data_name, self.string_type)
        out = self.token_embedding_layer(sequences)
   
        if self.abs_pos:
            out = self.positional_encoding(out)
        out = self.input_dropout(out)

        if self.abs_pos:
            mask = torch.zeros(batch_size, sequence_len, sequence_len, self.nhead, device=out.device)
        else:
            # relational positional encoding
            distance_squares = torch.abs(torch.arange(sequence_len).unsqueeze(0) - torch.arange(sequence_len).unsqueeze(1))
            distance_squares[distance_squares > self.max_len] = self.max_len
            distance_squares = distance_squares.unsqueeze(0).repeat(batch_size, 1, 1)
            distance_squares = distance_squares.to(out.device)
            mask = self.distance_embedding_layer(distance_squares)
            
        mask = mask.permute(0, 3, 1, 2)

        #
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1).to(out.device)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.reshape(-1, sequence_len, sequence_len)

        #
        
        key_padding_mask = sequences == TOKEN2ID[PAD_TOKEN]

        out = out.transpose(0, 1)
        out = self.transformer(out, mask, key_padding_mask)
        out = out.transpose(0, 1)

        #
        logits = self.generator(out)

        return logits

    def decode(self, num_samples, max_len, device):
        '''
        sequential generation
        '''
        TOKEN2ID = token_to_id_mol(self.data_name, self.string_type)
        node_type_list = NODE_TOKENS_DICT[self.data_name]
        node_types = [NODE_TYPE_DICT[node_type] for node_type in node_type_list]
        node_type_indices = [TOKEN2ID[node_type] for node_type in node_types]
        edge_types = BOND_TYPE_DICT.values()
        edge_type_indices = [TOKEN2ID[bond_type] for bond_type in edge_types]
        edge_indices = [value for key, value in TOKEN2ID.items() if type(key) is tuple]
        sequences = torch.LongTensor([[TOKEN2ID[BOS_TOKEN]] for _ in range(num_samples)]).to(device)
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        for index in tqdm(range(max_len), "generation"):
            # TODO: fix mask
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
            
            if index in [0, 1]:
                target_logits = node_type_logits
            else:
                target_logits = torch.where(is_node_type, edge_logits.T, node_type_logits.T).T
                target_logits = torch.where(is_edge, edge_type_logits.T, target_logits.T).T
                target_logits = torch.where(is_edge_type, node_type_edge_logits.T, target_logits.T).T
            
            # TODO: fix masked_logits
            # masked_logits = torch.where(mask, node_type_logits.T, edge_type_logits.T).T
            
            preds = Categorical(logits=target_logits).sample()
            preds[ended] = TOKEN2ID[PAD_TOKEN]
            sequences = torch.cat([sequences, preds.unsqueeze(1)], dim=1)

            ended = torch.logical_or(ended, preds == TOKEN2ID[EOS_TOKEN])

        return sequences