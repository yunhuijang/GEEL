import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
from tqdm import tqdm
import math
import os
import torch.nn.functional as F

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from data.tokens import TOKENS_DICT, TOKENS_DICT_DIFF, TOKENS_DICT_FLATTEN, TOKENS_DICT_SEQ, token_to_id
from data.mol_tokens import TOKENS_DICT_FLATTEN_MOL, TOKENS_DICT_MOL, TOKENS_DICT_SEQ_MOL, token_to_id_mol, id_to_token_mol, PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, NODE_TOKENS_DICT
from data.data_utils import NODE_TYPE_DICT, BOND_TYPE_DICT
from model.trans_generator import TokenEmbedding


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbeddingFeature(nn.Module):
    # TODO: token embedding eimension
    def __init__(self, vocab_size, emb_size, learn_pos, max_len, string_type, data_name, bw, num_nodes):
        super(TokenEmbeddingFeature, self).__init__()
        # self.num_nodes = int((-1+ math.sqrt(1 + 4*2*(vocab_size - 3))) / 2)
        self.num_nodes = num_nodes
        self.bw = bw
        self.emb_size = emb_size
        self.learn_pos = learn_pos
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embedding_numnode = nn.Embedding(self.num_nodes+4, emb_size)
        self.embedding_diff = nn.Embedding(self.bw+4, emb_size)
        self.data_name = data_name
        
        # max_len+2: for eos / bos token
        self.positional_embedding = nn.Parameter(torch.randn([1, max_len+2, emb_size]))

        self.string_type = string_type
        if self.string_type == 'adj_list':
            self.mol_tokens = TOKENS_DICT_MOL[self.data_name]
        elif self.string_type == 'adj_list_diff':
            self.mol_tokens = TOKENS_DICT_MOL[self.data_name]
        elif self.string_type in ['adj_flatten', 'adj_flatten_sym', 'bwr']:
            self.mol_tokens = TOKENS_DICT_FLATTEN_MOL[self.data_name]
        elif self.string_type in ['adj_seq', 'adj_seq_rel']:
            self.mol_tokens = TOKENS_DICT_SEQ_MOL[self.data_name]
        else:
            assert "No token type", False
        self.ID2TOKEN = id_to_token_mol(self.mol_tokens)
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
        if self.string_type in ['adj_flatten', 'adj_flatten_sym', 'bwr']:
            x = self.embedding(token_sequences) * math.sqrt(self.emb_size)
        elif self.string_type in ['adj_seq', 'adj_seq_rel']:
            x = self.embedding(token_sequences) * math.sqrt(self.emb_size)
        elif self.string_type in ['adj_list_diff', 'adj_list']:
            ID2TOKEN = id_to_token_mol(self.mol_tokens)
            t1, t2 = self.split_nodes(ID2TOKEN, token_sequences, device=token_sequences.device)
            # m = max(max(torch.flatten(t1)).item(), max(torch.flatten(t2)).item())
            x1 = self.embedding_numnode(t1) * math.sqrt(self.emb_size)
            if self.string_type == 'adj_list':
                x2 = self.embedding_numnode(t2) * math.sqrt(self.emb_size)
            elif self.string_type == 'adj_list_diff':
                x2 = self.embedding_diff(t2) * math.sqrt(self.emb_size)
            x = x1 + x2
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
    
# model that generates node and edge feature sequences (masking based on adj sequence)
class TransGeneratorFeature(nn.Module):
    '''
    without tree information (only string)
    '''
    
    def __init__(
        self, num_layers, emb_size, nhead, dim_feedforward, 
        input_dropout, dropout, max_len, string_type, learn_pos, abs_pos, 
        data_name, bw, num_nodes
    ):
        super(TransGeneratorFeature, self).__init__()
        self.nhead = nhead
        self.data_name = data_name
        self.string_type = string_type
        
        if self.string_type == 'adj_list':
            self.tokens = TOKENS_DICT[self.data_name]
            self.mol_tokens = TOKENS_DICT_MOL[self.data_name]
        elif self.string_type == 'adj_list_diff':
            self.tokens = TOKENS_DICT_DIFF[self.data_name]
            self.mol_tokens = TOKENS_DICT_MOL[self.data_name]
        elif self.string_type in ['adj_flatten', 'adj_flatten_sym', 'bwr']:
            self.tokens = TOKENS_DICT_FLATTEN[self.data_name]
            self.mol_tokens = TOKENS_DICT_FLATTEN_MOL[self.data_name]
        elif self.string_type in ['adj_seq', 'adj_seq_rel']:
            self.tokens = TOKENS_DICT_SEQ[self.data_name]
            self.mol_tokens = TOKENS_DICT_SEQ_MOL[self.data_name]
        else:
            assert False, "No token type"
        self.ID2TOKEN = id_to_token_mol(self.mol_tokens)
        
        self.TOKEN2ID = token_to_id_mol(self.data_name, self.string_type)
        self.learn_pos = learn_pos
        self.abs_pos = abs_pos
        self.max_len = max_len
        self.bw = bw
        self.num_nodes = num_nodes
        
        if self.abs_pos:
            self.positional_encoding = AbsolutePositionalEncoding(emb_size)
        
        self.token_embedding_layer_feature = TokenEmbeddingFeature(len(self.mol_tokens), emb_size, self.learn_pos, self.max_len, self.string_type, self.data_name, self.bw, self.num_nodes)
        self.token_embedding_layer = TokenEmbedding(len(self.tokens), emb_size, self.learn_pos, self.max_len, self.string_type, self.data_name, self.bw, self.num_nodes)
        self.input_dropout = nn.Dropout(input_dropout)
        
        #
        self.distance_embedding_layer = nn.Embedding(max_len + 1, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #
        self.generator = nn.Linear(emb_size, len(self.mol_tokens))
        

    def forward(self, sequences):
        # for training of qm9 / zinc
        if isinstance(sequences, tuple):
            adj_sequences = sequences[0]
            sequences = sequences[1]
        
        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)
        TOKEN2ID = token_to_id_mol(self.data_name, self.string_type)

        out = self.token_embedding_layer_feature(sequences)
        # if self.is_joint_adj:
        #     adj_out = self.token_embedding_layer(adj_sequences)
        #     adj_out = F.pad(input=adj_out, pad=(0, 0, 0, 1, 0, 0), value=0, mode='constant')
        #     out = adj_out + out
    
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

    def decode(self, num_samples, adj_list_sequences, max_len, device):
        '''
        sequential generation
        '''
        # ID2TOKEN = t(self.tokens)
        TOKEN2ID = token_to_id(self.data_name, self.string_type)
        TOKEN2IDFEA = token_to_id_mol(self.data_name, self.string_type)
        node_type_list = NODE_TOKENS_DICT[self.data_name]
        node_ids = [NODE_TYPE_DICT[node_type] for node_type in node_type_list]
        sequences = torch.LongTensor([[TOKEN2IDFEA[BOS_TOKEN]] for _ in range(num_samples)]).to(device)
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        node_indices = [TOKEN2IDFEA[node_type] for node_type in node_ids]
        bond_indices = [TOKEN2IDFEA[bond_type] for bond_type in BOND_TYPE_DICT.values()]
        special_token_indices = [TOKEN2IDFEA[token] for token in [PAD_TOKEN, EOS_TOKEN, BOS_TOKEN]]
        for index in tqdm(range(adj_list_sequences.shape[1]), "feature generation"):
            if ended.all():
                break
            current_adj_chars = adj_list_sequences[:,index]
            if index == 0:
                node_mask = torch.ones((len(current_adj_chars),), dtype=torch.bool).to(device)
            else:
                # node_mask: if node -> 1 else -> 0
                node_mask = torch.where(current_adj_chars == TOKEN2ID[0], True, False).to(device)
            end_mask = torch.where((current_adj_chars ==  TOKEN2ID[PAD_TOKEN]) | (current_adj_chars ==  TOKEN2ID[EOS_TOKEN]), True, False).to(device)
            ended = torch.logical_or(ended, end_mask)
            logits = self(sequences)
            
            # node_logits: mask edges / edge_logits: mask nodes
            node_logits = logits[:, -1].clone().detach()
            node_logits[:, bond_indices] = float("-inf")
            node_logits[:, special_token_indices] = float("-inf")
            edge_logits = logits[:, -1].clone().detach()
            edge_logits[:, node_indices] = float("-inf")
            edge_logits[:, special_token_indices] = float("-inf")
            
            masked_logits = torch.where(node_mask, node_logits.T, edge_logits.T).T
            preds = Categorical(logits=masked_logits).sample()
            preds[ended] = TOKEN2IDFEA[PAD_TOKEN]
            sequences = torch.cat([sequences, preds.unsqueeze(1)], dim=1)
        # set EOS
        sequences[range(len(sequences)), sequences.argmin(axis=1)] = TOKEN2IDFEA[EOS_TOKEN]
            

        return sequences