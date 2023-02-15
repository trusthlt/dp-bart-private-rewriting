import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.autoencoders.general_classes import Autoencoder_RNN, Autoencoder_Transformer, GeneralModelConfig


class CustomModelConfig(GeneralModelConfig):
    '''
    Description
    -----------
    Configuration class for custom model parameters that are not part of the
    framework.

    Parameters
    ----------
    custom_config_list : ``list``, A list of arguments provided by
                         `--custom_model_arguments` in CMD.
    '''
    def __init__(self, custom_config_list=[], **kwargs):
        super().__init__(**kwargs)
        self.custom_config_list = custom_config_list
        # For example:
        # self.alpha = custom_config_list[0]
        # self.beta = custom_config_list[1]
        # self.gamma = custom_config_list[2]


class CustomModel_RNN(Autoencoder_RNN):
    '''
    Description
    -----------
    Model class for a custom model that is not available as a pre-packaged
    module of the framework.

    Parameters
    ----------
    config : ``CustomModelConfig``, A class of configurations for the custom
             model, can be provided through CMD with
             `--custom_model_arguments`.
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self):
        '''
        Parameters
        ----------
        inputs : Dictionary of input values with key-value pairs:
            input_ids : `torch.LongTensor` (batch_size X seq_len)
            lengths : `torch.LongTensor` (batch_size)
            teacher_forcing_ratio : `float`

        Return
        ------
            z : `torch.DoubleTensor` (batch_size X seq_len-1 X vocab_size+4),
                An output tensor of logits, predicting the original input
                sequence (not including the initial SOS token). Output
                prediction space includes the 4 extra tokens (SOS/EOS/PAD/UNK).
        '''
        raise NotImplementedError


class CustomModel_Transformer(Autoencoder_Transformer):
    '''
    Description
    -----------
    Model class for a custom model that is not available as a pre-packaged
    module of the framework.

    Parameters
    ----------
    config : ``CustomModelConfig``, A class of configurations for the custom
             model, can be provided through CMD with
             `--custom_model_arguments`.
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, **inputs):
        '''
        Parameters
        ----------
        inputs : Dictionary of input values with key-value pairs:
            input_ids : `torch.LongTensor` (batch_size X seq_len)
            attention_mask : `torch.LongTensor` (batch_size X seq_len)

        Return
        ------
            z : `torch.LongTensor` (batch_size X seq_len-1 X vocab_size),
                An output tensor of token predictions for the original input
                sequence (not including the initial SOS token).
        '''
        raise NotImplementedError
