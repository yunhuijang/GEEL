import math
import torch
import numpy as np
import networkx as nx
import sentencepiece as spm

from data.orderings import bw_from_adj
from data.data_utils import unflatten_forward, train_data_to_string
from evaluation.evaluation import check_generated_samples


from torch_geometric.datasets import MNISTSuperpixels

from scipy.stats import entropy

for dataset_name in ['GDSS_com']:
# for dataset_name in ['GDSS_com']:
    for string_type in ['adj_flatten']:
        # check_generated_samples('GDSS_com')
        check_generated_samples(dataset_name, string_type)
        # train_data_to_string(dataset_name, string_type)
    # for string_type in ['adj_seq']:
        # spm.SentencePieceTrainer.Train(f"--input=samples/string/{dataset_name}/{string_type}.txt --model_prefix=resource/tokenizer/{dataset_name}/{string_type}_{vocab_size} --vocab_size={vocab_size} --model_type=bpe --character_coverage=1.0")