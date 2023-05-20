from dataload import DPRewriteDataset
from models.downstream_models.bert_downstream import BertDownstream
from utils import decode_rewritten, EarlyStopping
from models.autoencoders.adept import ADePT, ADePTModelConfig
from models.autoencoders.dp_bart import DPBart, DPBartModelConfig
from models.autoencoders.custom import CustomModel_RNN, CustomModel_Transformer, CustomModelConfig
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
from tqdm import tqdm
from copy import deepcopy
import os
import tempfile
from abc import ABC, abstractmethod
from settings import Settings
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sacrebleu.metrics import BLEU
from bert_score import BERTScorer
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'
import pdb
from utils import load_neurons_for_pruning, determine_neurons_to_prune, add_neurons_to_prune

import warnings
warnings.filterwarnings('ignore')


class Experiment(ABC):

    def __init__(self, ss:Settings):

        # General vars and directories
        self.seed = ss.args.seed
        self.local = ss.args.local
        self.local_iter_size = ss.args.local_iter_size
        self.mode = ss.args.mode

        self.exp_output_dir = ss.exp_output_dir
        self.exp_dump_dir = ss.exp_dump_dir
        self.checkpoint_dir = ss.checkpoint_dir

        self.asset_dir = ss.args.asset_dir
        self.embed_dir_unprocessed = ss.args.embed_dir_unprocessed
        self.embed_dir_processed = ss.embed_dir_processed

        self.dataset_name = ss.args.dataset
        self.custom_train_path = ss.args.custom_train_path
        self.custom_valid_path = ss.args.custom_valid_path
        self.custom_test_path = ss.args.custom_test_path

        self.last_checkpoint_path = ss.args.last_checkpoint_path
        self.save_unique_checkpoint_every_epoch =\
            ss.args.save_unique_checkpoint_every_epoch

        self.length_threshold = ss.args.length_threshold
        self.custom_preprocessor = ss.args.custom_preprocessor
        self.data_split_cutoff = ss.args.data_split_cutoff
        self.iteration_cutoff = ss.args.iteration_cutoff

        if self.local:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')

        # Hyperparameters (general)
        self.model = ss.args.model.lower()
        self.model_type = ss.args.model_type
        self.max_seq_len = ss.args.max_seq_len
        self.optim_type = ss.args.optim_type
        self.epochs = ss.args.epochs
        self.batch_size = ss.args.batch_size
        self.learning_rate = ss.args.learning_rate
        self.weight_decay = ss.args.weight_decay
        self.early_stop = ss.args.early_stopping
        self.patience = ss.args.patience

        self.pruning = ss.args.pruning
        self.prune_finetune_k = ss.args.prune_finetune_k
        self.pruning_index_path = ss.args.pruning_index_path
        self.gradually_increase_pruning = ss.args.gradually_increase_pruning

        self.early_stopping = EarlyStopping(self.patience)

        self.two_optimizers = ss.args.two_optimizers
        self.optimizer = None
        self.enc_optimizer = None
        self.dec_optimizer = None
        self.loss = None

        # Hyperparameters (specific to models)
        self.transformer_type = ss.args.transformer_type
        self.train_teacher_forcing_ratio = ss.args.train_teacher_forcing_ratio
            # only for transformer-based models
        self.hidden_size = ss.args.hidden_size
        self.enc_out_size = ss.args.enc_out_size
            # general for experiments that add DP module after encoder outputs
        self.embed_type = ss.args.embed_type
        self.vocab_size = ss.args.vocab_size
            # for experiments with non-HF-based tokenizers
        self.embed_size = ss.args.embed_size

        self.custom_model_arguments = ss.args.custom_model_arguments

        # Private parameters (not all necessary, depending on DP module):
        self.private = ss.args.private
        self.epsilon = ss.args.epsilon
        self.delta = ss.args.delta
        self.clipping_constant = ss.args.clipping_constant
        self.norm_ord = ss.args.l_norm
        self.dp_module = ss.args.dp_module
        self.dp_mechanism = ss.args.dp_mechanism
        self.discretize = ss.args.discretize
        if (self.dp_mechanism == 'gaussian' or self.dp_mechanism == 'analytic_gaussian') and self.norm_ord == 1:
            print(f"\n+++ WARNING: Using {self.dp_mechanism} noise with norm order {self.norm_ord}. +++\n")

        # Additional settings
        self.no_clipping = ss.args.no_clipping

        self.save_initial_model = ss.args.save_initial_model

        self.prepend_labels = ss.args.prepend_labels

        self.train_ratio = ss.args.train_ratio

        self.downstream_test_data = ss.args.downstream_test_data

        # General variables for experiments
        self.trainable_params = 0
        self.train_losses = []
        self.valid_losses = []

        # Variables for evaluation metrics
        self.bleu = BLEU()
        if self.device == torch.device('cuda'):
            self.run_bert_score = True
            self.bert_scorer = BERTScorer(
                lang='en', rescale_with_baseline=True,
                batch_size=4)
        else:
            self.run_bert_score = False

        self.temp_train_file_original = None
        self.temp_train_file_preds = None
        self.temp_valid_file_original = None
        self.temp_valid_file_preds = None

        # Write configuration and various stats to json files for documentation
        self.stats = {}

        config = {key: value for key, value in ss.args.__dict__.items()
                  if not key.startswith('__') and not callable(key)}
        with open(os.path.join(self.exp_output_dir, 'config.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    @abstractmethod
    def _load_checkpoint(self):
        pass

    @abstractmethod
    def train_iteration(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def plot_learning_curve(self):
        pass

    @abstractmethod
    def run_experiment(self):
        pass


class PretrainExperiment(Experiment):
    def __init__(self, ss: Settings):
        super().__init__(ss)
        self.dataset = DPRewriteDataset(
                self.dataset_name, self.asset_dir, self.checkpoint_dir,
                self.max_seq_len, self.batch_size, mode=self.mode,
                embed_type=self.embed_type, train_ratio=self.train_ratio,
                embed_size=self.embed_size,
                embed_dir_processed=self.embed_dir_processed,
                embed_dir_unprocessed=self.embed_dir_unprocessed,
                transformer_type=self.transformer_type,
                vocab_size=self.vocab_size,
                model_type=self.model_type, private=self.private,
                prepend_labels=self.prepend_labels,
                length_threshold=self.length_threshold,
                data_split_cutoff=self.data_split_cutoff,
                custom_preprocessor=self.custom_preprocessor,
                local=self.local,
                custom_train_path=self.custom_train_path,
                custom_valid_path=self.custom_valid_path,
                custom_test_path=self.custom_test_path,
                downstream_test_data=self.downstream_test_data
                )
        self.dataset.load_and_process()

        print('Initializing model...')
        self._init_model()

    def _init_model_config(self):
        # Setting the padding index
        if self.model_type == 'rnn':
            pad_idx = self.dataset.preprocessor.word2idx[
                self.dataset.preprocessor.PAD]
        else:  # 'transformer'
            pad_idx = self.dataset.preprocessor.tokenizer.pad_token_id

        self.pad_idx = pad_idx

        # Preparing the general model configuration
        general_config_dict = {
            'max_seq_len': self.max_seq_len, 'batch_size': self.batch_size,
            'mode': self.mode, 'local': self.local, 'device': self.device,
            'hidden_size': self.hidden_size,
            'enc_out_size': self.enc_out_size,
            'embed_size': self.embed_size, 'pad_idx': pad_idx,
            'transformer_type': self.transformer_type,
            'private': self.private, 'epsilon': self.epsilon,
            'delta': self.delta, 'norm_ord': self.norm_ord,
            'clipping_constant': self.clipping_constant,
            'dp_mechanism': self.dp_mechanism,
            'pruning': self.pruning,
            'pruning_index_path': self.pruning_index_path,
            'experiment_output_dir': self.exp_output_dir}

        # Preparing the specific model configuration class
        model_config = self._get_specific_model_config(general_config_dict)
        return model_config

    def _get_specific_model_config(self, general_config_dict):
        if self.model == 'adept':
            specific_config_dict = {
                'pretrained_embeddings': self.dataset.preprocessor.embeds,
                'vocab_size': self.dataset.preprocessor.vocab_size,
                'no_clipping': self.no_clipping
                }
            specific_config = ADePTModelConfig(
                **general_config_dict, **specific_config_dict
                )
        elif self.model == 'dp_bart':
            specific_config_dict = {
                'dp_module': self.dp_module,
                'no_clipping': self.no_clipping,
                'discretize': self.discretize
                }
            specific_config = DPBartModelConfig(
                **general_config_dict, **specific_config_dict
                )
        elif self.model in ['custom_rnn', 'custom_transformer']:
            specific_config_dict = {
                'custom_config_list': self.custom_model_arguments}
            specific_config = CustomModelConfig(
                **general_config_dict, **specific_config_dict
                )
        else:
            raise NotImplementedError
        return specific_config

    def _get_model_type(self):
        if self.model == 'adept':
            model_type = ADePT
        elif self.model == 'dp_bart':
            model_type = DPBart
        elif self.model == 'custom_rnn':
            model_type = CustomModel_RNN
        elif self.model == 'custom_transformer':
            model_type = CustomModel_Transformer
        else:
            raise NotImplementedError
        return model_type

    def _init_model(self):
        model_config = self._init_model_config()
        model_type = self._get_model_type()
        model = model_type(model_config)
        self.model = model.to(self.device)

        num_params = sum(p.numel()
                         for p in model.parameters() if p.requires_grad)
        print(f"Num parameters in model: {num_params,}")
        self.stats['num_params'] = num_params

        if self.optim_type == 'adam':
            optimizer = optim.Adam
        elif self.optim_type == 'sgd':
            optimizer = optim.SGD
        else:
            raise Exception('Incorrect optimizer type specified.')

        if self.two_optimizers:
            self.enc_optimizer = optimizer(
                self.model.encoder.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay)
            self.dec_optimizer = optim.Adam(
                self.model.decoder.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay)
        else:
            self.optimizer = optimizer(self.model.parameters(),
                                       lr=self.learning_rate,
                                       weight_decay=self.weight_decay)

        mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
        mem = mem_params + mem_bufs  # in bytes
        print("Estimated non-peak memory usage of model (MBs):", mem / 1000000)

        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def _load_checkpoint(self):
        '''
        Load existing checkpoint of a model and stats dict, if available.
        Stats dict only loaded if there is an existing checkpoint.
        '''
        try:
            mod_name = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
            checkpoint = torch.load(mod_name, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.two_optimizers:
                self.enc_optimizer.load_state_dict(
                    checkpoint['enc_optimizer_state_dict'])
                self.dec_optimizer.load_state_dict(
                    checkpoint['dec_optimizer_state_dict'])
            else:
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
            loaded_epoch = checkpoint['checkpoint_epoch'] + 1
                # Restart training from the next epoch
            early_stopping_counter = checkpoint['checkpoint_early_stopping']
            print(f"Loaded model from epoch {loaded_epoch} with early stopping counter at {early_stopping_counter}.")

            try:
                stats_path = os.path.join(self.exp_output_dir, 'stats.json')
                with open(stats_path, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            except:
                print("Could not load existing stats dictionary.")

        except:
            print("Could not load checkpointed model, starting from scratch...")
            loaded_epoch = 0
            early_stopping_counter = 0

        return loaded_epoch, early_stopping_counter

    def train_iteration(self, epoch):
        epoch_loss = 0

        if self.local:
            iter_size = self.local_iter_size
        else:
            iter_size = len(self.dataset.train_iterator)

        self.model.train()
        for idx, batch in tqdm(enumerate(self.dataset.train_iterator)):
            if self.local:
                if idx == iter_size:
                    break

            if self.iteration_cutoff is not None and idx == self.iteration_cutoff:
                break

            if self.model_type == 'rnn':
                encoder_input_ids = batch[0]
                lengths = batch[1]
                encoder_input_ids = encoder_input_ids.to(self.device)
                inputs = {'input_ids': encoder_input_ids, 'lengths': lengths,
                          'teacher_forcing_ratio': self.train_teacher_forcing_ratio}
                tgt = deepcopy(encoder_input_ids)[:, 1:].reshape(-1)
            else:
                encoder_input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                inputs = {'input_ids': encoder_input_ids,
                          'attention_mask': attention_mask}
                tgt = deepcopy(encoder_input_ids)[:, 1:].reshape(-1)

            if self.two_optimizers:
                self.enc_optimizer.zero_grad()
                self.dec_optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()

            loss = 0

            outputs = self.model(**inputs)

            loss = self.loss(outputs, tgt)

            loss.backward()

            if self.two_optimizers:
                torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(),
                                               max_norm=1)
                torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(),
                                               max_norm=1)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=1)

            if self.two_optimizers:
                self.enc_optimizer.step()
                self.dec_optimizer.step()
            else:
                self.optimizer.step()

            epoch_loss += loss.item()

            if idx == 0:
                preds = torch.max(outputs, dim=1).indices.view(
                        encoder_input_ids.shape[0],
                        encoder_input_ids.shape[1] - 1)

                decoded_text = decode_rewritten(
                        preds[0].unsqueeze(0),
                        self.dataset.preprocessor,
                        remove_special_tokens=False,
                        model_type=self.model_type)[0]
                original = decode_rewritten(
                        encoder_input_ids[0][1:].unsqueeze(0),
                        self.dataset.preprocessor,
                        remove_special_tokens=False,
                        model_type=self.model_type)[0]

                print("TRAIN ORIGINAL: ", original)
                print("TRAIN PRED: ", decoded_text)
                self.stats[f'sample_original_ep{epoch}_train'] = original
                self.stats[f'sample_pred_ep{epoch}_train'] = decoded_text

        return epoch_loss / iter_size

    def evaluate(self, epoch, final=False):
        epoch_loss = 0

        if self.local:
            iter_size = 1
        else:
            iter_size = len(self.dataset.valid_iterator)

        self.model.eval()
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.dataset.valid_iterator)):
                if self.local:
                    if idx == iter_size:
                        break

                if self.iteration_cutoff is not None and idx == self.iteration_cutoff:
                    break

                if self.model_type == 'rnn':
                    encoder_input_ids = batch[0]
                    lengths = batch[1]
                    encoder_input_ids = encoder_input_ids.to(self.device)
                    inputs = {'input_ids': encoder_input_ids, 'lengths': lengths,
                              'teacher_forcing_ratio': 0.0}
                    tgt = deepcopy(encoder_input_ids)[:, 1:].reshape(-1)
                else:
                    encoder_input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    inputs = {'input_ids': encoder_input_ids,
                              'attention_mask': attention_mask}
                    tgt = deepcopy(encoder_input_ids)[:, 1:].reshape(-1)

                if self.two_optimizers:
                    self.enc_optimizer.zero_grad()
                    self.dec_optimizer.zero_grad()
                else:
                    self.optimizer.zero_grad()

                loss = 0

                outputs = self.model(**inputs)

                loss = self.loss(outputs, tgt)
                loss = loss.item()

                epoch_loss += loss

                if idx == 0 and not final:
                    preds = torch.max(outputs, dim=1).indices.view(
                            encoder_input_ids.shape[0],
                            encoder_input_ids.shape[1] - 1)

                    decoded_text = decode_rewritten(
                            preds[0].unsqueeze(0),
                            self.dataset.preprocessor,
                            remove_special_tokens=False,
                            model_type=self.model_type)[0]
                    original = decode_rewritten(
                            encoder_input_ids[0][1:].unsqueeze(0),
                            self.dataset.preprocessor,
                            remove_special_tokens=False,
                            model_type=self.model_type)[0]

                    print("VALID ORIGINAL: ", original)
                    print("VALID PRED: ", decoded_text)
                    self.stats[f'sample_original_ep{epoch}_valid'] = original
                    self.stats[f'sample_pred_ep{epoch}_valid'] = decoded_text

                if final:
                    preds = torch.max(outputs, dim=1).indices.view(
                            encoder_input_ids.shape[0],
                            encoder_input_ids.shape[1] - 1)

                    decoded_text = decode_rewritten(
                            preds, self.dataset.preprocessor,
                            remove_special_tokens=True,
                            model_type=self.model_type)
                    original = decode_rewritten(
                            encoder_input_ids, self.dataset.preprocessor,
                            remove_special_tokens=True,
                            model_type=self.model_type)

                    for batch_idx in range(len(decoded_text)):
                        with open(self.temp_valid_file_preds, 'a', encoding='utf-8') as f:
                            f.write(decoded_text[batch_idx])
                            f.write('\n')
                        with open(self.temp_valid_file_original, 'a', encoding='utf-8') as f:
                            f.write(original[batch_idx])
                            f.write('\n')

        return epoch_loss / iter_size

    def train(self, loaded_epoch=0, early_stopping_counter=0):

        self.early_stopping.counter = early_stopping_counter
        for epoch in range(loaded_epoch, self.epochs):

            start_time = time.time()
            train_loss = self.train_iteration(epoch)
            valid_loss = self.evaluate(epoch, final=False)
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            self.plot_learning_curve()
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\tVal. Loss: {valid_loss:.3f}')

            # Saving checkpoint
            early_stop = self._save_checkpoint_and_early_stopping(
                epoch, valid_loss, early_stop=self.early_stop)
            if early_stop:
                break

            if self.pruning and self.gradually_increase_pruning:
                counter = epoch + 1
                self.model.model.add_neurons_to_prune(counter)

            # Updating stats dictionary
            self.stats[f'pretrain_epoch_mins_{epoch}'] = epoch_mins
            self.stats[f'pretrain_epoch_secs_{epoch}'] = epoch_secs
            self.stats[f'pretrain_train_loss_{epoch}'] = train_loss
            self.stats[f'pretrain_valid_loss_{epoch}'] = valid_loss

            # Saving stats dictionary
            self._save_stats_dict()

        # Only calculating BERTScore once at the end, and if running on GPU,
        # since it takes longer
        self.calculate_evaluation_metrics(
            epoch, bert_score=self.run_bert_score)

        # Saving stats dictionary one last time
        self._save_stats_dict()

    def _save_stats_dict(self):
        with open(os.path.join(self.exp_output_dir, 'stats.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=4)

    def _save_checkpoint_and_early_stopping(self, epoch, valid_loss, early_stop=True):
        if self.save_unique_checkpoint_every_epoch:
            checkpoint_name = os.path.join(self.checkpoint_dir, f'checkpoint_{epoch}.pt')
        else:
            checkpoint_name = os.path.join(self.checkpoint_dir, f'checkpoint.pt')
        if self.two_optimizers:
            checkpoint_dict = {
                'checkpoint_epoch': epoch,
                'checkpoint_early_stopping': self.early_stopping.counter,
                'model_state_dict': self.model.state_dict(),
                'enc_optimizer_state_dict': self.enc_optimizer.state_dict(),
                'dec_optimizer_state_dict': self.dec_optimizer.state_dict()
                }
        else:
            checkpoint_dict = {
                'checkpoint_epoch': epoch,
                'checkpoint_early_stopping': self.early_stopping.counter,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }

        # Early stopping
        if early_stop:
            self.early_stopping(valid_loss, checkpoint_dict, checkpoint_name)
            return self.early_stopping.early_stop
        else:
            torch.save(checkpoint_dict, checkpoint_name)
            return False

    def _set_up_evaluation_files(self):
        self.temp_valid_file_original = os.path.join(
                self.exp_output_dir, 'temp_valid_original.txt')
        self.temp_valid_file_preds = os.path.join(
                self.exp_output_dir, 'temp_valid_preds.txt')
        with open(self.temp_valid_file_original, 'w', encoding='utf-8') as f:
            f.write('valid original\n')
        with open(self.temp_valid_file_preds, 'w', encoding='utf-8') as f:
            f.write('valid preds\n')

    def calculate_evaluation_metrics(self, epoch, bert_score=False):
        print("Calculating evaluation metrics...")
        print("Performing final evaluation...")
        valid_loss = self.evaluate(epoch, final=True)
        valid_hyps, valid_refs = self._get_refs_and_hyps(
                self.temp_valid_file_preds, self.temp_valid_file_original)

        print("Calculating BLEU scores...")
        valid_bleu = self._calculate_bleu_score(valid_hyps, valid_refs)

        print("BLEU Valid:", valid_bleu)
        self.stats[f'bleu_ep{epoch}_valid'] = valid_bleu.score

        if bert_score:
            print("\nCalculating BERTScore...")
            valid_bert_res = self._calculate_bert_score(valid_hyps, valid_refs)
            print(f"BERTScore Valid (P): {valid_bert_res[0]:.2f}")
            print(f"BERTScore Valid (R): {valid_bert_res[1]:.2f}")
            print(f"BERTScore Valid (F1): {valid_bert_res[2]:.2f}")

            self.stats[f'bertscore_P_ep{epoch}_valid'] = valid_bert_res[0]
            self.stats[f'bertscore_R_ep{epoch}_valid'] = valid_bert_res[1]
            self.stats[f'bertscore_F1_ep{epoch}_valid'] = valid_bert_res[2]

    def _get_refs_and_hyps(self, preds_file, original_file):
        with open(preds_file, 'r', encoding='utf-8') as f:
            hyps = [x.strip() for x in f]
            hyps = hyps[1:]
        with open(original_file, 'r', encoding='utf-8') as f:
            refs = [x.strip() for x in f]
            refs = [refs[1:]]
        return hyps, refs

    def _calculate_bleu_score(self, hyps, refs):
        return self.bleu.corpus_score(hyps, refs)

    def _calculate_bert_score(self, hyps, refs):
        P, R, F1 = self.bert_scorer.score(hyps, refs)
        P = P.mean().item()
        R = R.mean().item()
        F1 = F1.mean().item()
        return (P, R, F1)

    def plot_learning_curve(self):
        '''
        Result png figures are saved in the log directory.
        '''
        fig, ax = plt.subplots(num=1, clear=True)
        fig.suptitle('Model Learning Curve')

        epochs = list(range(len(self.train_losses)))
        ax.plot(epochs, self.train_losses, 'o-', markersize=2, color='b',
                label='Train')
        ax.plot(epochs, self.valid_losses, 'o-', markersize=2, color='c',
                label='Validation')
        ax.set(xlabel='Epoch', ylabel='Pretrain Loss')
        ax.legend()

        plt.savefig(os.path.join(self.exp_output_dir, 'learning_curve.png'))

    def run_experiment(self):
        # Load an existing model checkpoint, if available
        loaded_epoch, early_stopping_counter = self._load_checkpoint()

        if self.save_initial_model and loaded_epoch == 0:
            # For convenient comparison of non-pretrained models
            print("Saving initial checkpoint of model...")
            self._save_checkpoint_and_early_stopping(-1, np.inf,
                                                     early_stop=False)

        # Setting up files for later evaluation of outputs
        self._set_up_evaluation_files()

        self.train(loaded_epoch=loaded_epoch,
                   early_stopping_counter=early_stopping_counter)


class RewriteExperiment(Experiment):
    def __init__(self, ss: Settings):
        super().__init__(ss)
        self.rewritten_dataset_dir = None

        # Whether to include original dataset in the rewritten dataframe
        self.include_original = ss.args.include_original

        self.dataset = DPRewriteDataset(
                self.dataset_name, self.asset_dir, self.checkpoint_dir,
                self.max_seq_len, self.batch_size, mode=self.mode,
                embed_type=self.embed_type, embed_size=self.embed_size,
                embed_dir_processed=self.embed_dir_processed,
                embed_dir_unprocessed=self.embed_dir_unprocessed,
                transformer_type=self.transformer_type,
                vocab_size=self.vocab_size,
                model_type=self.model_type, private=self.private,
                prepend_labels=self.prepend_labels,
                length_threshold=self.length_threshold,
                data_split_cutoff=self.data_split_cutoff,
                custom_preprocessor=self.custom_preprocessor, local=self.local,
                last_checkpoint_path=self.last_checkpoint_path,
                custom_train_path=self.custom_train_path,
                custom_valid_path=self.custom_valid_path,
                custom_test_path=self.custom_test_path,
                downstream_test_data=self.downstream_test_data
                )
        self.dataset.load_and_process()

        print('Initializing model...')
        self._init_model()

    def _init_model_config(self):
        # Setting the padding index
        if self.model_type == 'rnn':
            pad_idx = self.dataset.preprocessor.word2idx[
                self.dataset.preprocessor.PAD]
        else:  # 'transformer'
            pad_idx = self.dataset.preprocessor.tokenizer.pad_token_id

        self.pad_idx = pad_idx

        # Preparing the general model configuration
        general_config_dict = {
            'max_seq_len': self.max_seq_len, 'batch_size': self.batch_size,
            'mode': self.mode, 'local': self.local, 'device': self.device,
            'hidden_size': self.hidden_size,
            'enc_out_size': self.enc_out_size,
            'embed_size': self.embed_size, 'pad_idx': pad_idx,
            'transformer_type': self.transformer_type,
            'private': self.private, 'epsilon': self.epsilon,
            'delta': self.delta, 'norm_ord': self.norm_ord,
            'clipping_constant': self.clipping_constant,
            'dp_mechanism': self.dp_mechanism,
            'pruning': self.pruning,
            'pruning_index_path': self.pruning_index_path,
            'experiment_output_dir': self.exp_output_dir}

        # Preparing the specific model configuration class
        model_config = self._get_specific_model_config(general_config_dict)
        return model_config

    def _get_specific_model_config(self, general_config_dict):
        if self.model == 'adept':
            specific_config_dict = {
                'pretrained_embeddings': self.dataset.preprocessor.embeds,
                'vocab_size': self.dataset.preprocessor.vocab_size,
                'no_clipping': self.no_clipping
                }
            specific_config = ADePTModelConfig(
                **general_config_dict, **specific_config_dict
                )
        elif self.model == 'dp_bart':
            specific_config_dict = {
                'dp_module': self.dp_module,
                'no_clipping': self.no_clipping
                }
            specific_config = DPBartModelConfig(
                **general_config_dict, **specific_config_dict
                )
        elif self.model in ['custom_rnn', 'custom_transformer']:
            specific_config_dict = {
                'custom_config_list': self.custom_model_arguments}
            specific_config = CustomModelConfig(
                **general_config_dict, **specific_config_dict
                )
        else:
            raise NotImplementedError
        return specific_config

    def _get_model_type(self):
        if self.model == 'adept':
            model_type = ADePT
        elif self.model == 'dp_bart':
            model_type = DPBart
        elif self.model == 'custom_rnn':
            model_type = CustomModel_RNN
        elif self.model == 'custom_transformer':
            model_type = CustomModel_Transformer
        else:
            raise NotImplementedError
        return model_type

    def _init_model(self):
        model_config = self._init_model_config()
        model_type = self._get_model_type()
        model = model_type(model_config)
        self.model = model.to(self.device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Num parameters in model: {num_params,}")

        mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
        mem = mem_params + mem_bufs  # in bytes
        print("Estimated non-peak memory usage of model (MBs):", mem / 1000000)

        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.last_checkpoint_path,
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded pretrained model from {self.last_checkpoint_path}.")

    def evaluate(self, splits):
        with torch.no_grad():
            for split_name, iterator in splits.items():

                rewritten_df = pd.DataFrame(
                    columns=["label", "text", "original_label", "original_text"],
                    index=range(len(iterator.dataset)))

                epoch_loss = 0

                for idx, batch in tqdm(enumerate(iterator)):

                    if self.model_type == 'rnn':
                        encoder_input_ids = batch[0]
                        lengths = batch[1]
                        true_labels = batch[2]
                        encoder_input_ids = encoder_input_ids.to(self.device)
                        inputs = {'input_ids': encoder_input_ids, 'lengths': lengths,
                                  'teacher_forcing_ratio': 0.0}
                        tgt = deepcopy(encoder_input_ids)[:, 1:].reshape(-1)
                    else:
                        encoder_input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        true_labels = batch['labels'].to(self.device)
                        inputs = {'input_ids': encoder_input_ids,
                                  'attention_mask': attention_mask}
                        tgt = deepcopy(encoder_input_ids)[:, 1:].reshape(-1)

                    loss = 0
                    outputs = self.model(**inputs)

                    if self.model_type == 'rnn':
                        loss = self.loss(outputs, tgt)
                        loss = loss.item()
                        outputs_reshaped = outputs.view(
                                encoder_input_ids.shape[0],
                                encoder_input_ids.shape[1] - 1, outputs.shape[-1])
                        preds = torch.max(outputs_reshaped, dim=2).indices
                    else:
                        preds = outputs

                    if self.prepend_labels:
                        predicted_labels = preds[:, 0]
                        preds = preds[:, 1:]
                        encoder_input_ids = encoder_input_ids[:, 2:]
                        output_labels = decode_rewritten(
                            predicted_labels.unsqueeze(0),
                            self.dataset.preprocessor,
                            remove_special_tokens=False,
                            labels=True, model_type=self.model_type)[0]
                        true_labels = decode_rewritten(
                            true_labels.unsqueeze(0),
                            self.dataset.preprocessor,
                            remove_special_tokens=False,
                            labels=True, model_type=self.model_type)[0]
                    else:
                        if self.model_type == 'rnn':
                            output_labels = true_labels
                        else:
                            true_labels = [
                                self.dataset.preprocessor.lab_int2str[lab.item()]
                                for lab in true_labels
                                ]
                            output_labels = true_labels

                    decoded_text = decode_rewritten(
                            preds, self.dataset.preprocessor,
                            remove_special_tokens=True,
                            model_type=self.model_type)
                    original = decode_rewritten(
                            encoder_input_ids, self.dataset.preprocessor,
                            remove_special_tokens=True,
                            model_type=self.model_type)

                    for batch_idx in range(len(decoded_text)):
                        current_data_idx = (idx * self.batch_size) + batch_idx
                        if decoded_text[batch_idx] == '':
                            decoded_text[batch_idx] = ' '
                        current_data_point = decoded_text[batch_idx]
                        current_data_original = original[batch_idx]
                        current_data_label = output_labels[batch_idx]
                        current_data_original_label = true_labels[batch_idx]
                        rewritten_df["text"].loc[current_data_idx] =\
                            current_data_point
                        rewritten_df["original_text"].loc[current_data_idx] =\
                            current_data_original
                        rewritten_df["label"].loc[current_data_idx] = \
                            current_data_label
                        rewritten_df["original_label"].loc[current_data_idx] = \
                            current_data_original_label

                    epoch_loss += loss
                final_loss = epoch_loss / len(iterator)
                print(f"{split_name} set: | Rewrite loss: {final_loss} |")

                if not self.include_original:
                    rewritten_df_out = rewritten_df.drop('original_text', 1)
                    rewritten_df_out = rewritten_df_out.drop('original_label', 1)
                else:
                    rewritten_df_out = rewritten_df

                rewritten_df_out.to_csv(
                        os.path.join(
                            self.rewritten_dataset_dir,
                            f"rewritten_{split_name}.csv"),
                        header=False,
                        index=False)

                self.calculate_evaluation_metrics(
                    rewritten_df, split_name, bert_score=self.run_bert_score)

    def train(self):
        pass

    def train_iteration(self):
        pass

    def plot_learning_curve(self):
        pass

    def calculate_evaluation_metrics(self, rewritten_df, split_name,
                                     bert_score=False):
        print("Calculating evaluation metrics...")
        hyps, refs = self._get_refs_and_hyps(rewritten_df)

        print("Calculating BLEU scores...")
        bleu_score = self._calculate_bleu_score(hyps, refs)

        print(f"BLEU {split_name}:", bleu_score)

        self.stats[f'bleu_{split_name}'] = bleu_score.score

        if bert_score:
            print("\nCalculating BERTScore...")
            bert_res = self._calculate_bert_score(hyps, refs)

            print(f"BERTScore {split_name} (P): {bert_res[0]:.2f}")
            print(f"BERTScore {split_name} (R): {bert_res[1]:.2f}")
            print(f"BERTScore {split_name} (F1): {bert_res[2]:.2f}")

            self.stats[f'bertscore_P_{split_name}'] = bert_res[0]
            self.stats[f'bertscore_R_{split_name}'] = bert_res[1]
            self.stats[f'bertscore_F1_{split_name}'] = bert_res[2]

    def _get_refs_and_hyps(self, rewritten_df):
        hyps = rewritten_df['text'].tolist()
        refs = [rewritten_df['original_text'].tolist()]
        return hyps, refs

    def _calculate_bleu_score(self, hyps, refs):
        return self.bleu.corpus_score(hyps, refs)

    def _calculate_bert_score(self, hyps, refs):
        P, R, F1 = self.bert_scorer.score(hyps, refs)
        P = P.mean().item()
        R = R.mean().item()
        F1 = F1.mean().item()
        return (P, R, F1)

    def run_experiment(self):
        # Preparing dataset directory
        self.rewritten_dataset_dir = os.path.join(
            self.exp_dump_dir, self.dataset_name + "_rewritten")
        if not os.path.exists(self.rewritten_dataset_dir):
            os.makedirs(self.rewritten_dataset_dir)

        # Loading the pre-trained model
        self._load_checkpoint()

        splits = {'train': self.dataset.train_iterator,
                  'valid': self.dataset.valid_iterator,
                  'test': self.dataset.test_iterator}
        splits = {split_name: iterator
                  for split_name, iterator in splits.items()
                  if iterator is not None}

        self.evaluate(splits)

        # Saving stats dictionary
        with open(os.path.join(self.exp_output_dir, 'stats.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=4)


class DownstreamExperiment(Experiment):
    def __init__(self, ss: Settings):
        super().__init__(ss)
        self.test_only = ss.args.test_only

        self.transformer_type = ss.args.transformer_type

        self.train_accs = []
        self.valid_accs = []
        self.train_precs = []
        self.valid_precs = []
        self.train_recs = []
        self.valid_recs = []
        self.train_f1s = []
        self.valid_f1s = []

        self.dataset = DPRewriteDataset(
                self.dataset_name, self.asset_dir, self.checkpoint_dir,
                self.max_seq_len, self.batch_size, mode=self.mode,
                train_ratio=self.train_ratio, embed_type=self.embed_type,
                embed_size=self.embed_size,
                embed_dir_processed=self.embed_dir_processed,
                embed_dir_unprocessed=self.embed_dir_unprocessed,
                vocab_size=self.vocab_size,
                model_type=self.model_type, private=self.private,
                prepend_labels=self.prepend_labels,
                transformer_type=self.transformer_type,
                length_threshold=self.length_threshold,
                data_split_cutoff=self.data_split_cutoff,
                custom_preprocessor=self.custom_preprocessor,
                local=self.local,
                custom_train_path=self.custom_train_path,
                custom_valid_path=self.custom_valid_path,
                custom_test_path=self.custom_test_path,
                downstream_test_data=self.downstream_test_data
                )

        self.dataset.load_and_process()

        self.output_dim = self.dataset.preprocessor.num_labels

        print('Initializing model...')
        self._init_model()

    def _init_model(self):
        if self.model == 'bert_downstream':
            expmod = BertDownstream
        else:
            raise NotImplementedError

        model = expmod(output_dim=self.output_dim, avg_hs_output=False,
                       dropout=0.8, transformer_type=self.transformer_type,
                       hidden_dim=768, local=self.local,
                       device=self.device)
        self.model = model.to(self.device)

        if self.optim_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        elif self.optim_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.learning_rate,
                                       weight_decay=self.weight_decay)
        else:
            raise Exception('Incorrect optimizer type specified.')

        num_params = sum(p.numel() for p in model.parameters()
                         if p.requires_grad)
        print(f"Num parameters in model: {num_params,}")
        self.trainable_params = num_params

        mem_params = sum([param.nelement()*param.element_size()
                         for param in model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size()
                       for buf in model.buffers()])
        mem = mem_params + mem_bufs  # in bytes
        print("Estimated non-peak memory usage of model (MBs):", mem / 1000000)

        if self.output_dim == 1:
            self.binary_classification = True
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.binary_classification = False
            self.loss = nn.CrossEntropyLoss()

    def _load_checkpoint(self):
        '''
        Load existing checkpoint of a model and stats dict, if available.
        Stats dict only loaded if there is an existing checkpoint.
        '''
        try:
            mod_name = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
            checkpoint = torch.load(mod_name, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])
            loaded_epoch = checkpoint['checkpoint_epoch'] + 1
                # restart training from the next one
            early_stopping_counter = checkpoint['checkpoint_early_stopping']
            self.train_accs = checkpoint['train_accs']
            self.train_f1s = checkpoint['train_f1s']
            self.train_losses = checkpoint['train_losses']
            self.valid_accs = checkpoint['valid_accs']
            self.valid_f1s = checkpoint['valid_f1s']
            self.valid_losses = checkpoint['valid_losses']
            print(f"Loaded model from epoch {loaded_epoch} with early stopping counter at {early_stopping_counter}.")

            try:
                stats_path = os.path.join(self.exp_output_dir, 'stats.json')
                with open(stats_path, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            except:
                print("Could not load existing stats dictionary.")

        except:
            print("Could not load checkpointed model, starting from scratch...")
            loaded_epoch = 0
            early_stopping_counter = 0

        return loaded_epoch, early_stopping_counter

    def train_iteration(self, epoch):
        epoch_loss = 0
        epoch_acc = 0
        epoch_prec = 0
        epoch_rec = 0
        epoch_f1 = 0

        if self.local:
            iter_size = self.local_iter_size
        else:
            iter_size = len(self.dataset.train_iterator)

        self.model.train()
        for idx, batch in tqdm(enumerate(self.dataset.train_iterator)):
            if self.local:
                if idx == iter_size:
                    break

            if self.iteration_cutoff is not None and idx == self.iteration_cutoff:
                break

            input_ids = batch['input_ids']
            input_ids = input_ids.to(self.device)

            attention_mask = batch['attention_mask']
            attention_mask = attention_mask.to(self.device)

            labels = batch['labels']
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            loss = 0

            logits = self.model(input_ids, attention_mask)

            loss = self.loss(logits, labels)

            loss.backward()

            self.optimizer.step()

            acc, prec, rec, f1 = self.calculate_accuracy(logits, labels)

            epoch_loss += loss.item()

            if idx == 0:
                sample_doc = self.dataset.preprocessor.tokenizer.decode(
                    input_ids[0, :], skip_special_tokens=True)
                sample_label = labels[0].item()
                sample_pred = logits.argmax(dim=1)[0].item()
                print("TRAIN DOC: ", sample_doc)
                print("TRAIN LABEL: ", sample_label)
                print("TRAIN PRED: ", sample_pred)
                self.stats[f'sample_doc_ep{epoch}_train'] = sample_doc
                self.stats[f'sample_label_ep{epoch}_train'] = sample_label
                self.stats[f'sample_pred_ep{epoch}_train'] = sample_pred

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_prec += prec
            epoch_rec += rec
            epoch_f1 += f1

        return epoch_loss / iter_size, epoch_acc / iter_size,\
            epoch_prec / iter_size, epoch_rec / iter_size, epoch_f1 / iter_size

    def evaluate(self, epoch, test=False):
        epoch_loss = 0
        epoch_acc = 0
        epoch_prec = 0
        epoch_rec = 0
        epoch_f1 = 0

        if self.local:
            iter_size = self.local_iter_size
        else:
            iter_size = len(self.dataset.valid_iterator)\
                if not test else len(self.dataset.test_iterator)

        if test:
            iterator = self.dataset.test_iterator
            partition_name = 'TEST'
        else:
            iterator = self.dataset.valid_iterator
            partition_name = 'VALID'

        self.model.eval()
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(iterator)):
                if self.local:
                    if idx == iter_size:
                        break

                if self.iteration_cutoff is not None and idx == self.iteration_cutoff:
                    break

                input_ids = batch['input_ids']
                input_ids = input_ids.to(self.device)

                attention_mask = batch['attention_mask']
                attention_mask = attention_mask.to(self.device)

                labels = batch['labels']
                labels = labels.to(self.device)

                loss = 0

                logits = self.model(input_ids, attention_mask)

                loss = self.loss(logits, labels)

                acc, prec, rec, f1 = self.calculate_accuracy(logits, labels)

                epoch_loss += loss.item()

                if idx == 0 and not test:
                    sample_doc = self.dataset.preprocessor.tokenizer.decode(
                        input_ids[0, :], skip_special_tokens=True)
                    sample_label = labels[0].item()
                    sample_pred = logits.argmax(dim=1)[0].item()
                    print(f"{partition_name} DOC: ", sample_doc)
                    print(f"{partition_name} LABEL: ", sample_label)
                    print(f"{partition_name} PRED: ", sample_pred)
                    self.stats[f'sample_doc_ep{epoch}_'
                               f'{partition_name.lower()}'] = sample_doc
                    self.stats[f'sample_label_ep{epoch}_'
                               f'{partition_name.lower()}'] = sample_label
                    self.stats[f'sample_pred_ep{epoch}_'
                               f'{partition_name.lower()}'] = sample_pred

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                epoch_prec += prec
                epoch_rec += rec
                epoch_f1 += f1

        return epoch_loss / iter_size, epoch_acc / iter_size,\
            epoch_prec / iter_size, epoch_rec / iter_size, epoch_f1 / iter_size

    def calculate_accuracy(self, logits, labels):
        preds = logits.argmax(dim=1)

        majority = False
        random = False
        if majority:
            print("Calculating majority baseline...")
            maj_val = torch.mode(
                self.dataset.train_iterator.dataset['labels']).values.item()
            preds = torch.ones(logits.shape[0], dtype=torch.int64) * maj_val
            preds = preds.to(self.device)
        else:
            if random:
                print("Calculating random baseline...")
                preds = torch.randint(0, self.dataset.preprocessor.num_labels, (labels.shape))
                preds = preds.to(self.device)

        acc = torch.sum(preds == labels) / len(preds)
        results = precision_recall_fscore_support(
            labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        prec, rec, f1, _ = results

        return acc, prec, rec, f1

    def plot_learning_curve(self):
        '''
        Result png figures are saved in the log directory.
        '''
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, clear=True)
        fig.suptitle('Model Learning Curve')

        epochs = list(range(len(self.train_losses)))
        ax1.plot(epochs, self.train_losses, 'o-', markersize=2, color='b',
                 label='Train')
        ax1.plot(epochs, self.valid_losses, 'o-', markersize=2, color='c',
                 label='Validation')
        ax1.set(ylabel='Loss')

        ax2.plot(epochs, self.train_f1s, 'o-', markersize=2, color='b',
                 label='Train')
        ax2.plot(epochs, self.valid_f1s, 'o-', markersize=2, color='c',
                 label='Validation')
        ax2.set(xlabel='Epoch', ylabel='F1 Score')
        ax1.legend()

        plt.savefig(os.path.join(self.exp_output_dir, 'learning_curve.png'))

    def train(self, loaded_epoch=0, early_stopping_counter=0):

        self.early_stopping.counter = early_stopping_counter
        for epoch in range(loaded_epoch, self.epochs):

            start_time = time.time()
            train_loss, train_A, train_P, train_R, train_F1 =\
                self.train_iteration(epoch)
            valid_loss, valid_A, valid_P, valid_R, valid_F1 =\
                self.evaluate(epoch, test=False)
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.train_accs.append(train_A)
            self.valid_accs.append(valid_A)
            self.train_precs.append(train_P)
            self.valid_precs.append(valid_P)
            self.train_recs.append(train_R)
            self.valid_recs.append(valid_R)
            self.train_f1s.append(train_F1)
            self.valid_f1s.append(valid_F1)

            self.plot_learning_curve()
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m'
                  f'{epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\tVal. Loss: {valid_loss:.3f}')
            print(f'\tTrain Acc: {train_A:.3f}')
            print(f'\tVal. Acc: {valid_A:.3f}')
            print(f'\tTrain Prec: {train_P:.3f}')
            print(f'\tVal. Prec: {valid_P:.3f}')
            print(f'\tTrain Rec: {train_R:.3f}')
            print(f'\tVal. Rec: {valid_R:.3f}')
            print(f'\tTrain F1: {train_F1:.3f}')
            print(f'\tVal. F1: {valid_F1:.3f}')
            checkpoint_name = os.path.join(self.checkpoint_dir,
                                           'checkpoint.pt')
            checkpoint_dict = {
                'checkpoint_epoch': epoch,
                'checkpoint_early_stopping': self.early_stopping.counter,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_accs': self.train_accs,
                'train_f1s': self.train_f1s,
                'train_losses': self.train_losses,
                'valid_accs': self.valid_accs,
                'valid_f1s': self.valid_f1s,
                'valid_losses': self.valid_losses
                }
            self.stats[f'downstream_epoch_mins_{epoch}'] = epoch_mins
            self.stats[f'downstream_epoch_secs_{epoch}'] = epoch_secs
            self.stats[f'downstream_train_loss_{epoch}'] = train_loss
            self.stats[f'downstream_valid_loss_{epoch}'] = valid_loss
            self.stats[f'downstream_train_acc_{epoch}'] = train_A
            self.stats[f'downstream_valid_acc_{epoch}'] = valid_A
            self.stats[f'downstream_train_P_{epoch}'] = train_P
            self.stats[f'downstream_valid_P_{epoch}'] = valid_P
            self.stats[f'downstream_train_R_{epoch}'] = train_R
            self.stats[f'downstream_valid_R_{epoch}'] = valid_R
            self.stats[f'downstream_train_F1_{epoch}'] = train_F1
            self.stats[f'downstream_valid_F1_{epoch}'] = valid_F1

            # Saving stats dictionary
            with open(os.path.join(self.exp_output_dir, 'stats.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=4)

            if self.early_stop:
                self.early_stopping(valid_loss, checkpoint_dict,
                                    checkpoint_name)
                if self.early_stopping.early_stop:
                    break
            else:
                # If no early stopping, just save every epoch
                torch.save(checkpoint_dict, checkpoint_name)

    def output_results(self, test_loss, test_A, test_F1):
        if self.early_stop and self.early_stopping.best_score is not None:
            best_val_loss = -self.early_stopping.best_score
        else:
            best_val_loss = min(self.valid_losses)
        filepath = os.path.join(self.exp_output_dir, 'results.csv')
        epoch = self.valid_losses.index(best_val_loss)
        best_val_acc = self.valid_accs[epoch]
        best_val_f1 = self.valid_f1s[epoch]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('BestValidLoss,BestValidAcc,BestValidF1,BestValidEpoch,'
                    'TestLoss,TestAcc,TestF1,NumTrainableParams\n')
            f.write(f'{best_val_loss:.4f},{best_val_acc:.4f},'
                    f'{best_val_f1:.4f},{epoch},{test_loss:.4f},'
                    f'{test_A:.4f},{test_F1:.4f},{self.trainable_params}')

    def run_experiment(self):
        # Load an existing model checkpoint, if available
        loaded_epoch, early_stopping_counter = self._load_checkpoint()

        if not self.test_only:
            self.train(loaded_epoch=loaded_epoch,
                       early_stopping_counter=early_stopping_counter)
        else:
            print("Running downstream experiment in 'test_only' configuration.")
            if loaded_epoch == 0:
                raise Exception("Model checkpoint could not be loaded based on specified experiment name and results directory.")

        # Test and output results
        if not self.dataset.test_iterator:
            print("Test set not provided, setting test loss, accuracy and F1"
                  " as 0 in output results.")
            test_loss, test_A, test_F1 = 3*(0.0,)
        else:
            # Load the best model
            try:
                mod_name = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
                checkpoint = torch.load(mod_name, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except:
                print("Could not load checkpointed model for final evaluation, using model from last trained epoch...")

            test_loss, test_A, test_P, test_R, test_F1 =\
                self.evaluate(None, test=True)
        self.output_results(test_loss, test_A, test_F1)
        print("Final test loss:", test_loss)
        print("Final test accuracy:", test_A)
        print("Final test F1:", test_F1)
