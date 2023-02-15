import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from abc import abstractmethod


class GeneralModelConfig(object):
    def __init__(self, **kwargs):
        # General parameters
        self.max_seq_len = kwargs.pop("max_seq_len", 20)
        self.batch_size = kwargs.pop("batch_size", 32)
        self.mode = kwargs.pop("mode", 'pretrain')
        self.local = kwargs.pop("local", False)
        self.device = kwargs.pop("device", 'cpu')

        # RNN-based parameters
        self.hidden_size = kwargs.pop("hidden_size", 128)
        self.enc_out_size = kwargs.pop("enc_out_size", 128)
        self.embed_size = kwargs.pop("embed_size", 300)
        self.pad_idx = kwargs.pop("pad_idx", 0)

        # Transformer-based parameters
        self.transformer_type = kwargs.pop("transformer_type",
                                           "facebook/bart-base")

        # Privacy-related parameters
        self.private = kwargs.pop("private", False)
        self.epsilon = kwargs.pop("epsilon", 10000)
        self.delta = kwargs.pop("delta", 1e-7)
        self.norm_ord = kwargs.pop("norm_ord", 2)
        self.clipping_constant = kwargs.pop("clipping_constant", 1)
        self.dp_mechanism = kwargs.pop("dp_mechanism", 'laplace')

        # Pruning-related parameters
        self.pruning = kwargs.pop("pruning", False)
        self.pruning_index_path = kwargs.pop("pruning_index_path", '.')
        self.experiment_output_dir = kwargs.pop("experiment_output_dir", '.')


class Autoencoder(nn.Module):
    def __init__(self, max_seq_len=20, batch_size=32, mode='pretrain',
                 local=False, device='cpu'):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        # In case pre-training and rewriting models need to exhibit some
        # differences in behavior
        self.mode = mode

        self.local = local
        self.device = device

    @abstractmethod
    def forward(self):
        raise NotImplementedError


class Autoencoder_RNN(Autoencoder):
    def __init__(self, hidden_size=128, enc_out_size=128, vocab_size=20000,
                 embed_size=300, pad_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.enc_out_size = enc_out_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.pad_idx = pad_idx

    @abstractmethod
    def forward(self):
        raise NotImplementedError


class Autoencoder_Transformer(Autoencoder):
    def __init__(self, transformer_type='facebook/bart-base', **kwargs):
        super().__init__(**kwargs)
        self.transformer_type = transformer_type

    @abstractmethod
    def forward(self):
        raise NotImplementedError
