import torch
import torch.nn as nn
from torch.distributions import Categorical
import math

from data.tokens import map_tokens, token_to_id, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class LSTMGenerator(nn.Module):
    def __init__(
        self, emb_size, dropout, num_layers, string_type, dataset, vocab_size, is_token=False
    ):
        super(LSTMGenerator, self).__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.string_type = string_type
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.is_token = is_token
        
        self.tokens = map_tokens(self.dataset, self.string_type, self.vocab_size, self.is_token)

        self.TOKEN2ID = token_to_id(self.dataset, self.string_type, self.is_token, self.vocab_size)
        self.vocab_size = self.output_size = len(self.tokens)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.emb_size)
        self.lstm_layer = nn.LSTM(self.emb_size, self.emb_size,  
                                dropout=self.dropout, batch_first=True, num_layers=self.num_layers)
        self.linear_layer = nn.Linear(self.emb_size, self.output_size)


    def forward(self, sequences):
        batch_size = sequences.size(0)
        out = self.embedding_layer(sequences)
        out, hidden = self.lstm_layer(out)
        logits = self.linear_layer(out)

        return logits

    def decode(self, num_samples, max_len, device):
        sequences = torch.LongTensor([[self.TOKEN2ID[BOS_TOKEN]] for _ in range(num_samples)]).to(device)
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        for _ in range(max_len):
            if ended.all():
                break

            logits = self(sequences)
            preds = Categorical(logits=logits[:, -1]).sample()
            preds[ended] = self.TOKEN2ID[PAD_TOKEN]
            sequences = torch.cat([sequences, preds.unsqueeze(1)], dim=1)
            ended = torch.logical_or(ended, preds == self.TOKEN2ID[EOS_TOKEN])
        return sequences
