import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
from tqdm import tqdm
import numpy as np
from collections import deque
from torch.nn.functional import pad
from joblib import Parallel, delayed

from data.tokens import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, TOKENS_DICT, token_to_id


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class TreePositionalEncoding(nn.Module):
    def __init__(self, d_model, token2id, pos_type):
        super().__init__()
        self.d_model = d_model
        self.pos_type = pos_type
        self.pos_dict = {'1': (1,0,0,0), '2': (0,1,0,0), '3': (0,0,1,0), '4': (0,0,0,1)}
        self.token2id = token2id
        self.bos = self.token2id[BOS_TOKEN]
        self.eos = self.token2id[EOS_TOKEN]
        # self.zero = self.token2id['0']
        
    def map_pos_to_tensor(self, pos):
        result = []
        for char in pos:
            result.extend(self.pos_dict[char])
            
        return torch.tensor(result)
    
    def filter_row_string(self, row_string):
        # filter bos to eos
        l = [int(char) for char in [*row_string]]
        try:
            return l[1:l.index(self.eos)]
        except ValueError:
            return l[1:]
    
    def get_pe(self, row_string):
        l = self.filter_row_string(row_string)
        if len(l) == 0:
            return torch.zeros((1,1))
        elif len(l) > 3:
            str_queue = deque(l[0:4])
            pos_list = [1,2,3,4]
        else:
            str_queue = deque(l[0:len(l)])
            pos_list = list(range(1,len(l)+1))
            
        pos_queue = deque(pos_list)
        for i in np.arange(4,len(l),4):
            cur_string = l[i:i+4]
            cur_parent = str_queue.popleft()
            cur_parent_pos = pos_queue.popleft()
            while((cur_parent == self.token2id['0']) and (len(str_queue) > 0)):
                cur_parent = str_queue.popleft()
                cur_parent_pos = pos_queue.popleft()
            cur_pos = [cur_parent_pos*10+i for i in range(1,1+len(cur_string))]
            pos_list.extend(cur_pos)
            pos_queue.extend(cur_pos)
            str_queue.extend(cur_string)

        reverse_pos_list = [str(pos)[::-1] for pos in pos_list]
        tensor_pos_list = [self.map_pos_to_tensor(pos) for pos in reverse_pos_list]
        max_size = len(tensor_pos_list[-1])
        final_pos_list = [pad(pos, (0,max_size-len(pos))) for pos in tensor_pos_list]
        
        # return shape: sequence len * pe size
        return torch.stack(final_pos_list)
        
    
    def positional_embedding(self, x, pe_size, emb_size):
        embedding = nn.Linear(pe_size, emb_size)
        return embedding(x.float()) * math.sqrt(emb_size)
    
    
    def forward(self, x, org_x):
        # x shape: batch size * sequence len * emb size
        if x.size(1) == 1:
            return x
        pe_list= Parallel(n_jobs=8)(delayed(self.get_pe)(x_string) for x_string in org_x)
            # [self.get_pe(x_string) for x_string in org_x]
        max_str_length = max([pe.shape[0] for pe in pe_list])
        max_pe_length = max([pe.shape[1] for pe in pe_list])
        # top: bos / bottom: eos
        pe_tensor = torch.stack([pad(pe, (0, max_pe_length-pe.shape[1],1,max_str_length-pe.shape[0]+1)) for pe in pe_list])
        # generation (no eos)
        if pe_tensor.shape[1] != x.shape[1]:
            pe_tensor = torch.stack([pad(pe, (0, max_pe_length-pe.shape[1],1,max_str_length-pe.shape[0])) for pe in pe_list])
        
        if self.pos_type == 'pad':
            padder = torch.nn.ReplicationPad2d((0,x.shape[-1]-pe_tensor.shape[-1],0,0))
            pe = padder(pe_tensor.to(float)).to(x.device)
        elif self.pos_type == 'emb':
            pe = self.positional_embedding(pe_tensor, max_pe_length, self.d_model).to(x.device)
        
        x += pe
        return x


class SmilesGenerator(nn.Module):
    '''
    without tree information (only string)
    '''
    # TODO: add string positional encodings (absolute encoding, etc.)
    
    def __init__(
        self, num_layers, emb_size, nhead, dim_feedforward, 
        input_dropout, dropout, max_len, string_type, tree_pos, pos_type
    ):
        super(SmilesGenerator, self).__init__()
        self.nhead = nhead
        self.tokens = TOKENS_DICT[string_type]
        self.TOKEN2ID = token_to_id(self.tokens)
        self.string_type = string_type
        self.tree_pos = tree_pos
        self.pos_type = pos_type
        if string_type in ['group', 'bfs-deg-group']:
            self.max_len = int(max_len/4)
        else:
            self.max_len = max_len
        #
        self.token_embedding_layer = TokenEmbedding(len(self.tokens), emb_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.positional_encoding = TreePositionalEncoding(emb_size, self.TOKEN2ID, self.pos_type)
        #
        self.distance_embedding_layer = nn.Embedding(max_len + 1, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #
        self.generator = nn.Linear(emb_size, len(self.tokens))

    def forward(self, sequences):
        
        sequences = sequences
        
        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)
        TOKEN2ID = token_to_id(self.tokens)
        #
        out = self.token_embedding_layer(sequences)
        if self.tree_pos:
            out = self.positional_encoding(out, sequences)
        out = self.input_dropout(out)


        if self.tree_pos:
            mask = torch.zeros(batch_size, sequence_len, sequence_len, self.nhead, device=out.device)
        else:
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
        TOKEN2ID = token_to_id(self.tokens)
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
