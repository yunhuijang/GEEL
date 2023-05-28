import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
from tqdm import tqdm
import numpy as np
from collections import deque
from torch.nn.functional import pad
import math
from torch.nn.utils.rnn import pad_sequence
from time import time
import re

from data.tokens import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, TOKENS_DICT, token_to_id, id_to_token

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, learn_pos, max_len, string_type):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 3, emb_size)
        self.emb_size = emb_size
        self.learn_pos = learn_pos
        # max_len+2: for eos / bos token
        self.positional_embedding = nn.Parameter(torch.randn([1, max_len+2, emb_size]))

        self.string_type = string_type
        self.tokens = TOKENS_DICT[string_type]
        self.ID2TOKEN = id_to_token(self.tokens)
    
    def split_nodes(self, dictionary, input_tokens, device):
        mapping_tensor1 = torch.zeros(len(dictionary), dtype=torch.long, device=device)
        mapping_tensor2 = torch.zeros(len(dictionary), dtype=torch.long, device=device)

        for key, value in dictionary.items():
            if isinstance(value, tuple):
                mapping_tensor1[key] = value[0] + 3
                mapping_tensor2[key] = value[1] + 3
            else:
                mapping_tensor1[key] = key
                mapping_tensor2[key] = key

        sequence_tensor = input_tokens
        output_tokens1 = mapping_tensor1[sequence_tensor]
        output_tokens2 = mapping_tensor2[sequence_tensor]

        return output_tokens1, output_tokens2
        
    def forward(self, tokens):
        
        ID2TOKEN = id_to_token(TOKENS_DICT[self.string_type])

        t1, t2 = self.split_nodes(ID2TOKEN, tokens, device=tokens.device)
        x1 = self.embedding(t1) * math.sqrt(self.emb_size)
        x2 = self.embedding(t2) * math.sqrt(self.emb_size)
        x = (x1+x2)/2.0

        x_batch_size = x.shape[0]
        x_seq_len = x.shape[1]
        if self.learn_pos:
            pe = self.positional_embedding[:,:x_seq_len]
            pe_stack = torch.tile(pe, (x_batch_size, 1, 1))
            return x+pe_stack
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
    

class TransGenerator(nn.Module):
    '''
    without tree information (only string)
    '''
    
    def __init__(
        self, num_layers, emb_size, nhead, dim_feedforward, 
        input_dropout, dropout, max_len, string_type, learn_pos, abs_pos
    ):
        super(TransGenerator, self).__init__()
        self.nhead = nhead
        self.tokens = TOKENS_DICT[string_type]
        self.ID2TOKEN = id_to_token(self.tokens)
        self.string_type = string_type
        self.TOKEN2ID = token_to_id(self.string_type)
        self.learn_pos = learn_pos
        self.abs_pos = abs_pos
        self.max_len = max_len
        
        
        if self.abs_pos:
            self.positional_encoding = AbsolutePositionalEncoding(emb_size)
        
        self.token_embedding_layer = TokenEmbedding(len(self.tokens), emb_size, self.learn_pos, self.max_len, self.string_type)
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

        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)
        TOKEN2ID = token_to_id(self.string_type)

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
        # TODO: maybe need change afterwards
        TOKEN2ID = token_to_id(self.string_type)
        sequences = torch.LongTensor([[TOKEN2ID[BOS_TOKEN]] for _ in range(num_samples)]).to(device)
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        for _ in tqdm(range(max_len), "generation"):
            if ended.all():
                break
            logits = self(sequences)
            preds = Categorical(logits=logits[:, -1]).sample()
            preds[ended] = TOKEN2ID[PAD_TOKEN]
            sequences = torch.cat([sequences, preds.unsqueeze(1)], dim=1)

            ended = torch.logical_or(ended, preds == TOKEN2ID[EOS_TOKEN])

        return sequences