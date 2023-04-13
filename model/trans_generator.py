import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
from tqdm import tqdm

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
    def __init__(self, d_model, type='pad'):
        super().__init__()
        self.d_model = d_model
        
    
    def get_pe_tensor(self, tree, max_nodes, max_pe_length):
        pe_tensor = torch.stack([node.data for node in tree.nodes.values() if not node.is_root()])
        num_nodes = pe_tensor.shape[0]
        pe_length = pe_tensor.shape[1]
        padder = nn.ZeroPad2d((0,max_pe_length-pe_length,1,max_nodes-num_nodes))
        return padder(pe_tensor)
    
    # def positional_embedding(self, x, pe_size, emb_size):
    #     embedding = nn.Embedding(pe_size, emb_size)
    #     return embedding(x.long()) * math.sqrt(emb_size)
    
    def forward(self, x, trees):
        # -1: except for root
        max_nodes = max([len(tree.nodes) for tree in trees])
        max_pe_length = max([len(tree.nodes['0'].data) for tree in trees])
        pe_tensor = torch.stack([self.get_pe_tensor(tree, max_nodes, max_pe_length) for tree in trees])
        # 1. pad_pe_tensor: make pe to be same dim with x 
        padder = torch.nn.ReplicationPad2d((0,x.shape[-1]-pe_tensor.shape[-1],0,0))
        pad_pe_tensor = padder(pe_tensor).to(x.device)
        # pe = self.positional_embedding(pe_tensor, max_pe_length, self.d_model)
        x += pad_pe_tensor
        return x


class SmilesGenerator(nn.Module):
    def __init__(
        self, num_layers, emb_size, nhead, dim_feedforward, input_dropout, dropout, max_len, string_type, tree_pos
    ):
        super(SmilesGenerator, self).__init__()
        self.nhead = nhead
        self.tokens = TOKENS_DICT[string_type]
        self.string_type = string_type
        if string_type == 'group':
            self.max_len = int(max_len/4)
        else:
            self.max_len = max_len
        self.tree_pos = tree_pos
        #
        self.token_embedding_layer = TokenEmbedding(len(self.tokens), emb_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.positional_encoding = TreePositionalEncoding(emb_size)
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

        #
        out = self.token_embedding_layer(sequences)
        if self.tree_pos:
            out = self.positional_encoding(out, trees)
        out = self.input_dropout(out)

        #
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
        TOKEN2ID = token_to_id(self.tokens)
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
        # empty_tree = Tree()
        # empty_tree.create_node("root", "0")
        # cur_tree_list = [empty_tree for _ in range(num_samples)]
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        for _ in tqdm(range(max_len), "generation"):
            if ended.all():
                break
            
            # Parallel(n_jobs=8)(delayed(bfs_string_to_tree)(string) for string in sequences)
            # logits = self((sequences, cur_tree_list))
            logits = self(sequences)
            preds = Categorical(logits=logits[:, -1]).sample()
            preds[ended] = TOKEN2ID[PAD_TOKEN]
            sequences = torch.cat([sequences, preds.unsqueeze(1)], dim=1)
            # strings = [untokenize(sequence, self.string_type)[1][5:] for sequence in sequences.tolist()]
            # cur_tree_list = [bfs_string_to_tree(string) for string in strings]

            ended = torch.logical_or(ended, preds == TOKEN2ID[EOS_TOKEN])

        return sequences
