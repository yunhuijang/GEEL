from joblib.parallel import delayed
from networkx.readwrite.gml import Token
import torch
import torch.nn as nn
from torch.distributions import Categorical
import math

from tqdm import tqdm
from joblib import Parallel, delayed

from data.target_data import PAD_TOKEN, TOKENS, RING_START_TOKEN, RING_END_TOKENS, MAX_LEN, Data, get_id
from model.lr import PolynomialDecayLR

from time import time

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class EdgeLogitLayer(nn.Module):
    def __init__(self, emb_size, hidden_dim):
        super(EdgeLogitLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5
        self.linear0 = nn.Linear(emb_size, self.hidden_dim)
        self.linear1 = nn.Linear(emb_size, self.hidden_dim)

    def forward(self, x, sequences):
        batch_size = x.size(0)
        seq_len = x.size(1)
        out0 = self.linear0(x).view(batch_size, seq_len, self.hidden_dim)

        out1_ = self.linear1(x).view(batch_size, seq_len, self.hidden_dim)

        ring_start_mask = (sequences == get_id(RING_START_TOKEN))
        index_ = ring_start_mask.long().cumsum(dim=1)
        index_ = index_.masked_fill(~ring_start_mask, 0)
        
        out1 = torch.zeros(batch_size, len(RING_END_TOKENS) + 1, self.hidden_dim).to(out1_.device)
        out1.scatter_(dim=1, index=index_.unsqueeze(-1).repeat(1, 1, self.hidden_dim), src=out1_)
        out1 = out1[:, 1:]
        out1 = out1.permute(0, 2, 1)
        logits = self.scale * torch.bmm(out0, out1)

        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class BaseGenerator(nn.Module):
    def __init__(
        self, 
        num_layers, 
        emb_size, 
        nhead, 
        dim_feedforward, 
        input_dropout, 
        dropout, 
        disable_treeloc, 
        disable_graphmask, 
        disable_valencemask,
        enable_absloc
        ):
        super(BaseGenerator, self).__init__()
        self.nhead = nhead

        #
        self.token_embedding_layer = TokenEmbedding(len(TOKENS), emb_size)
        self.count_embedding_layer = TokenEmbedding(MAX_LEN, emb_size)
        
        #
        if enable_absloc:
            self.positional_encoding = PositionalEncoding(emb_size)

        self.input_dropout = nn.Dropout(input_dropout)
        
        #
        self.linear_loc_embedding_layer = nn.Embedding(MAX_LEN + 1, nhead)
        self.up_loc_embedding_layer = nn.Embedding(MAX_LEN + 1, nhead)
        self.down_loc_embedding_layer = nn.Embedding(MAX_LEN + 1, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #
        self.generator = nn.Linear(emb_size, len(TOKENS) - len(RING_END_TOKENS))
        self.ring_generator = EdgeLogitLayer(emb_size=emb_size, hidden_dim=emb_size)

        #
        self.disable_treeloc = disable_treeloc
        self.disable_graphmask = disable_graphmask
        self.disable_valencemask = disable_valencemask
        self.enable_absloc = enable_absloc

    def forward(self, batched_data):
        (
            sequences,
            count_sequences, 
            graph_mask_sequences,
            valence_mask_sequences,
            linear_loc_squares,
            up_loc_squares,
            down_loc_squares,
        ) = batched_data
        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)

        #
        out = self.token_embedding_layer(sequences)
        out += self.count_embedding_layer(count_sequences)

        if self.enable_absloc:
            out = self.positional_encoding(out)

        out = self.input_dropout(out)

        #
        mask = torch.zeros(batch_size, sequence_len, sequence_len, self.nhead, device=out.device)
        if not self.enable_absloc:
            mask += self.linear_loc_embedding_layer(linear_loc_squares)
            
        if not self.disable_treeloc:
            mask += self.up_loc_embedding_layer(up_loc_squares)
            mask += self.down_loc_embedding_layer(down_loc_squares)
        
        mask = mask.permute(0, 3, 1, 2)

        #
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1).to(out.device)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.reshape(-1, sequence_len, sequence_len)

        #
        key_padding_mask = sequences == get_id(PAD_TOKEN)

        out = out.transpose(0, 1)
        out = self.transformer(out, mask, key_padding_mask)
        out = out.transpose(0, 1)

        #
        logits0 = self.generator(out)
        logits1 = self.ring_generator(out, sequences)
        logits = torch.cat([logits0, logits1], dim=2)
        if not self.disable_graphmask:
            logits = logits.masked_fill(graph_mask_sequences, float("-inf"))
         
        if not self.disable_valencemask:
            logits = logits.masked_fill(valence_mask_sequences, float("-inf"))

        return logits

    def decode(self, num_samples, max_len, device):
        data_list = [Data() for _ in range(num_samples)]
        ended_data_list = []

        def _update_data(inp):
            data, id = inp
            data.update(id)
            return data

        for idx in range(max_len):
            if len(data_list) == 0:
                break

            feature_list = [data.featurize() for data in data_list]
            batched_data = Data.collate(feature_list)
            batched_data = [tsr.to(device) for tsr in batched_data]

            logits = self(batched_data)
            preds = Categorical(logits=logits[:, -1]).sample()

            data_list = [_update_data(pair) for pair in zip(data_list, preds.tolist())]
            ended_data_list += [data for data in data_list if data.ended]
            data_list = [data for data in data_list if not data.ended]

            if idx == max_len-1:
                for data in data_list:
                    data.error = "incomplete"
        
        data_list = data_list + ended_data_list

        return data_list

class CondGenerator(BaseGenerator):
    def __init__(
        self, 
        num_layers, 
        emb_size, 
        nhead, 
        dim_feedforward, 
        input_dropout, 
        dropout, 
        disable_treeloc, 
        disable_graphmask, 
        disable_valencemask,
        enable_absloc, 
        ):
        super(CondGenerator, self).__init__(
            num_layers, 
            emb_size, 
            nhead, 
            dim_feedforward, 
            input_dropout, 
            dropout, 
            disable_treeloc, 
            disable_graphmask, 
            disable_valencemask,
            enable_absloc, 
        )
        self.cond_embedding_layer = nn.Linear(1, emb_size)

    def forward(self, batched_mol_data, batched_cond_data):
        (
            sequences,
            count_sequences, 
            graph_mask_sequences,
            valence_mask_sequences,
            linear_loc_squares,
            up_loc_squares,
            down_loc_squares,
        ) = batched_mol_data

        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)

        #
        out = self.token_embedding_layer(sequences)
        out += self.count_embedding_layer(count_sequences)
        
        #
        out += self.cond_embedding_layer(batched_cond_data).view(batch_size, 1, -1)

        out = self.input_dropout(out)

        #
        
        mask = torch.zeros(batch_size, sequence_len, sequence_len, out.size(-1), device=out.device)
        if not self.enable_absloc:
            mask += self.linear_loc_embedding_layer(linear_loc_squares)

        if not self.disable_treeloc:
            mask += self.up_loc_embedding_layer(up_loc_squares)
            mask += self.down_loc_embedding_layer(down_loc_squares)
        
        mask = mask.permute(0, 3, 1, 2)

        #
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1).to(out.device)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.reshape(-1, sequence_len, sequence_len)

        #
        key_padding_mask = sequences == get_id(PAD_TOKEN)

        out = out.transpose(0, 1)
        out = self.transformer(out, mask, key_padding_mask)
        out = out.transpose(0, 1)

        #
        logits0 = self.generator(out)
        logits1 = self.ring_generator(out, sequences)
        logits = torch.cat([logits0, logits1], dim=2)
        if not self.disable_graphmask:
            logits = logits.masked_fill(graph_mask_sequences, float("-inf"))
         
        if not self.disable_valencemask:
            logits = logits.masked_fill(valence_mask_sequences, float("-inf"))

        return logits

    def decode(self, batched_cond_data, max_len, device):
        num_samples = batched_cond_data.size(0)
        data_list = [Data() for _ in range(num_samples)]
        ended_data_list = []

        data_idxs = list(range(num_samples))
        
        def _update_data(inp):
            data, id = inp
            data.update(id)
            return data

        for idx in range(max_len):
            if len(data_list) == 0:
                break

            feature_list = [data.featurize() for data in data_list]
            batched_mol_data = Data.collate(feature_list)
            batched_mol_data = [tsr.to(device) for tsr in batched_mol_data]

            logits = self(batched_mol_data, batched_cond_data[data_idxs])
            preds = Categorical(logits=logits[:, -1]).sample()

            data_list = [_update_data(pair) for pair in zip(data_list, preds.tolist())]
            ended_data_list += [data for data in data_list if data.ended]
            data_idxs = [data_idx for data_idx, data in zip(data_idxs, data_list) if not data.ended]
            data_list = [data for data in data_list if not data.ended]
            if idx == max_len-1:
                for data in data_list:
                    data.error = "incomplete"
        
        data_list = data_list + ended_data_list

        return data_list