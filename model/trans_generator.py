import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
from tqdm import tqdm
import math
from time import time
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from data.tokens import PAD_TOKEN, token_to_id, id_to_token, map_tokens
from model.lstm_generator import LSTMGenerator, TokenEmbedding


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
    
# model with pure Transformer (no masking in generation)
class TransGenerator(LSTMGenerator):
    '''
    without tree information (only string)
    '''
    
    def __init__(
        self, num_layers, emb_size, nhead, dim_feedforward, 
        input_dropout, dropout, max_len, string_type, learn_pos, abs_pos, 
        data_name, bw, num_nodes, is_token, vocab_size
    ):
        super(TransGenerator, self).__init__(emb_size, dropout, num_layers, string_type, data_name,
                                             vocab_size, num_nodes, max_len, bw, is_token, learn_pos)
        self.nhead = nhead
        self.data_name = data_name
        self.string_type = string_type
        self.is_token = is_token
        self.vocab_size = vocab_size
        self.tokens = map_tokens(self.data_name, self.string_type, self.vocab_size, self.is_token)
        self.ID2TOKEN = id_to_token(self.tokens)
        
        self.TOKEN2ID = token_to_id(self.data_name, self.string_type, self.is_token, self.vocab_size)
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
        
        key_padding_mask = sequences == self.TOKEN2ID[PAD_TOKEN]

        out = out.transpose(0, 1)
        out = self.transformer(out, mask, key_padding_mask)
        out = out.transpose(0, 1)

        #
        logits = self.generator(out)

        return logits

