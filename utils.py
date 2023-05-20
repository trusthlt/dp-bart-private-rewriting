import torch
from torch.utils.data import Dataset, TensorDataset
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import json
import os
import pdb

from math import exp, sqrt
from scipy.special import erf

import mpmath
from mpmath import mp

def calibrateAnalyticGaussianMechanism_precision(epsilon, delta, GS, tol = 1.e-12):
    """
    High-precision version of `calibrateAnalyticGaussianMechanism` function below.
    Should not be used for epsilon values above 5000.
    """

    if epsilon <= 1000:
        mp.dps = 500  # works for epsilon = 1000
    elif epsilon <= 2500 and epsilon > 1000:
        mp.dps = 1100  # works for epsilon = 2500
    else:
        mp.dps = 2200  # works for epsilon = 5000

    def Phi(t):
        return 0.5*(1.0 + mpmath.erf(t/mpmath.sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(mpmath.sqrt(epsilon*s)) - mpmath.exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-mpmath.sqrt(epsilon*s)) - mpmath.exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : mpmath.sqrt(1.0 + s/2.0) - mpmath.sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : mpmath.sqrt(1.0 + s/2.0) + mpmath.sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)

    sigma = alpha*GS/mpmath.sqrt(2.0*epsilon)

    return float(sigma)


def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol = 1.e-12):
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

    Source: https://github.com/BorjaBalle/analytic-gaussian-mechanism

    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)
    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)

    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma


def get_model_type(model):
    '''
    Given a specified model for an experiment, return the type of model ('rnn'
    or 'transformer')
    '''
    model_to_model_type = {
        'adept': 'rnn',
        'dp_bart': 'transformer',
        'custom_rnn': 'rnn',
        'custom_transformer': 'transformer',
        'bert_downstream': 'transformer'
        }
    if model in model_to_model_type.keys():
        model_type = model_to_model_type[model]
    else:
        raise Exception("Specified model not in current list of available "
                        "models.")
    return model_type


class EarlyStopping(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, checkpoint_dict, mod_name):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            torch.save(checkpoint_dict, mod_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of '
                  f'{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            checkpoint_dict['checkpoint_early_stopping'] = self.counter
            torch.save(checkpoint_dict, mod_name)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def decode_rewritten(rewritten, preprocessor, remove_special_tokens=True,
                     labels=False, model_type='transformer'):
    if model_type == 'rnn':
        decoded = decode_rewritten_rnn(
            rewritten, preprocessor,
            remove_special_tokens=remove_special_tokens, labels=labels)
    else:
        decoded = decode_rewritten_transformer(
            rewritten, preprocessor,
            remove_special_tokens=remove_special_tokens, labels=labels)

    return decoded


def decode_rewritten_rnn(rewritten, preprocessor, remove_special_tokens=True,
                         labels=False):
    '''
    rewritten: torch tensor size batch X max_seq_len-1, type int64
    preprocessor: preprocessing class from preprocessing.py
    remove_special_tokens: ignore <pad>, <unk>, <sos> and <eos> tokens

    decoded: list of strings, with predicted tokens separated by a space
    '''
    special_tokens = [0, 1, 2, 3]

    decoded = []
    for batch_idx in range(rewritten.shape[0]):
        batch = rewritten[batch_idx, :]
        if remove_special_tokens:
            decoded_batch = [preprocessor.idx2word[idx.item()] for idx in batch if idx not in special_tokens]
        else:
            decoded_batch = [preprocessor.idx2word[idx.item()] for idx in batch]
        decoded.append(decoded_batch)

    if not labels:
        decoded = [' '.join(batch) for batch in decoded]

    # For empty strings
    decoded = [doc if doc != '' else 'UNK' for doc in decoded]

    return decoded


def decode_rewritten_transformer(rewritten, preprocessor,
                                 remove_special_tokens=True, labels=False):
    '''
    rewritten: torch tensor size batch X max_seq_len-1, type int64
    preprocessor: preprocessing class from preprocessing.py
    remove_special_tokens: ignore special tokens according to huggingface's
                           tokenizer

    decoded: list of strings, with predicted tokens separated by a space
    '''
    decoded = preprocessor.tokenizer.batch_decode(
        rewritten, skip_special_tokens=remove_special_tokens)

    if labels:
        raise NotImplementedError

    return decoded


def determine_neurons_to_prune(model, device='cpu', out_path='.'):
    k_proj_weights = model.decoder.layers[0].encoder_attn.k_proj.weight
    v_proj_weights = model.decoder.layers[0].encoder_attn.v_proj.weight
    k_abs_sums = torch.sum(torch.abs(k_proj_weights), dim=0).to(device)
    v_abs_sums = torch.sum(torch.abs(v_proj_weights), dim=0).to(device)

    q = torch.tensor([0.25, 0.5, 0.75]).to(device)
    k_quantile_values = torch.quantile(k_abs_sums, q, dim=0, keepdim=True)
    k_threshold = k_quantile_values[0].item()

    v_quantile_values = torch.quantile(v_abs_sums, q, dim=0, keepdim=True)
    v_threshold = v_quantile_values[0].item()

    k_prune_neurons = torch.where(k_abs_sums < k_threshold)[0]
    v_prune_neurons = torch.where(v_abs_sums < v_threshold)[0]

    print(f"Pruned neurons per token (k-projection): {k_prune_neurons.shape[0]}")
    torch.save(k_prune_neurons, out_path)

    return k_prune_neurons, v_prune_neurons


def add_neurons_to_prune(model, previous_k_prune_neurons, device='cpu', out_path='.'):
    previous_k_prune_neurons = previous_k_prune_neurons.to(device)
    k_proj_weights = model.decoder.layers[0].encoder_attn.k_proj.weight

    original_indexes = torch.arange(768).to(device)
    combined = torch.cat((original_indexes, previous_k_prune_neurons))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]

    k_abs_sums = torch.sum(torch.abs(k_proj_weights), dim=0)
    k_abs_sums_smaller = k_abs_sums[difference]

    q = torch.tensor([0.25, 0.5, 0.75]).to(device)
    quantile_values = torch.quantile(k_abs_sums_smaller, q, dim=0, keepdim=True)
    threshold = quantile_values[0].item()

    k_prune_neurons_smaller_indexes = torch.where(
        k_abs_sums_smaller < threshold)[0]
    k_prune_neurons = difference[k_prune_neurons_smaller_indexes]

    all_k_prune_neurons = torch.sort(torch.cat((previous_k_prune_neurons, k_prune_neurons))).values
    print(f"Pruned another {k_prune_neurons.shape[0]} neurons per token (k-projection), total pruned now: {all_k_prune_neurons.shape[0]} per token")
    torch.save(all_k_prune_neurons, out_path)

    return all_k_prune_neurons


def load_neurons_for_pruning(in_path='.'):
    k_prune_neurons = torch.load(in_path)
    return k_prune_neurons


def non_intersection(tensor1, tensor2):
    combined = torch.cat((tensor1, tensor2))
    uniques, counts = combined.unique(return_counts=True)
    non_intersection = uniques[counts == 1]
    return non_intersection


def prepare_specific_experiment(ss, experiment='adept_l1norm_pretrain'):
    '''
    Sets up arguments to fit a given experiment, overwrites required default
    values to properly fit with the experiment.
    For more customizability, can leave 'experiment' as None and select one's
    own parameters for the experiment.
    E.g. 'adept'
    '''
    if experiment == 'adept_l1norm_pretrain':
        ss.args.mode = 'pretrain'
        ss.args.model = 'adept'
        ss.args.model_type = 'rnn'
        ss.args.no_clipping = False
        ss.args.prepend_labels = False
        ss.args.private = False
        ss.args.l_norm = 1
    if experiment == 'adept_l2norm_pretrain':
        ss.args.mode = 'pretrain'
        ss.args.model = 'adept'
        ss.args.model_type = 'rnn'
        ss.args.no_clipping = False
        ss.args.prepend_labels = False
        ss.args.private = False
        ss.args.l_norm = 2
    if experiment == 'adept_l1norm_rewrite':
        ss.args.mode = 'rewrite'
        ss.args.model = 'adept'
        ss.args.model_type = 'rnn'
        ss.args.prepend_labels = False
        ss.args.private = True
        ss.args.l_norm = 1
    if experiment == 'adept_l2norm_rewrite':
        ss.args.mode = 'rewrite'
        ss.args.model = 'adept'
        ss.args.model_type = 'rnn'
        ss.args.prepend_labels = False
        ss.args.private = True
        ss.args.l_norm = 2

    return ss
