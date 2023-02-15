from abc import ABC, abstractmethod
from datasets import Dataset, Value

import os
import copy
import json
import multiprocessing

import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from torchtext.data.utils import get_tokenizer
from gensim.models import KeyedVectors

from utils import NpEncoder


class Preprocessor(ABC):
    '''
    Description
    -----------
    Abstract class based on which built-in and custom preprocessors are
    prepared.

    Attributes
    ----------
    checkpoint_dir : str
    max_seq_len : int
    batch_size : int
    prepend_labels : bool
    length_threshold : int

    Methods
    -------
    process_data():
        Given a loaded dataset from HF or local path, output the preprocessed
        dataset.
    '''
    def __init__(self, checkpoint_dir, max_seq_len, batch_size,
                 prepend_labels, mode):
        self.checkpoint_dir = checkpoint_dir
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.prepend_labels = prepend_labels
        self.mode = mode

    @abstractmethod
    def process_data(self, data):
        '''
        Description
        -----------

        Given a loaded dataset from HF or local path, output the preprocessed
        dataset.

        Parameters
        ----------
        data : ``Dataset``, A dataset object from HF with at least 'label' and
               'text' columns.

        '''
        pass


class Preprocessor_for_RNN(Preprocessor):
    def __init__(self, embed_dir_processed, embed_dir_unprocessed,
                 vocab_size=20000, embed_type='glove', embed_size=300,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dir_processed = embed_dir_processed
        self.embed_dir_unprocessed = embed_dir_unprocessed

        self.vocab_size = vocab_size
        self.embed_type = embed_type
        self.embed_size = embed_size

        self.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        self.PAD = '<PAD>'
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.UNK = '<UNK>'

        if self.embed_type not in ['none']:
            self.vocab, self.embeds, self.word2idx, self.idx2word =\
                    self._make_vocab_and_embeds_files()
        else:
            self.vocab, self.embeds, self.word2idx, self.idx2word = (None,)*4

    def _make_vocab_and_embeds_files(self):
        '''
        Returns:
            vocab (np.ndarray): 1D array of strings, untrimmed vocabulary
            embeds (np.ndarray): 2D array, of untrimmed vocabulary size X
                                 self.embed_size
        '''
        vocab_file = os.path.join(
            self.embed_dir_processed,
            f'vocab_type{self.embed_type}_d{self.embed_size}_np.npy')
        embeds_file = os.path.join(
            self.embed_dir_processed,
            f'embeds_type{self.embed_type}_d{self.embed_size}_np.npy')

        try:
            with open(vocab_file, 'rb') as v_f:
                vocab = np.load(v_f)
            with open(embeds_file, 'rb') as e_f:
                embeds = np.load(e_f)
            print("Loaded vocabulary and embedding files...")
        except FileNotFoundError:
            print("Preparing vocabulary and embedding files...")
            if self.embed_type.lower() == 'glove':
                vocab, embeds = self.make_vocab_and_embeds_glove(
                    vocab_file, embeds_file)
            elif self.embed_type.lower() in ['word2vec', 'w2v']:
                vocab, embeds = self.make_vocab_and_embeds_w2v(
                    vocab_file, embeds_file)
            else:
                raise Exception(
                    "'embed_type' can only be 'glove', 'word2vec', or 'none'.")

        indexes = np.arange(vocab.size)
        word2idx = {}
        idx2word = {}
        for idx, word in zip(indexes, vocab):
            word2idx[word] = idx
            idx2word[idx] = word

        # Additionally adding <UNK> (will be reindexed later)
        word2idx[self.UNK] = len(indexes) + 1
        idx2word[len(indexes) + 1] = self.UNK

        return vocab, embeds, word2idx, idx2word

    def make_vocab_and_embeds_w2v(self, vocab_file, embeds_file):
        model = KeyedVectors.load_word2vec_format(self.embed_dir_unprocessed,
                                                  binary=True)

        vocab_np = np.array(model.index_to_key)
        embeds_np = model.vectors
        del model

        np.save(vocab_file, vocab_np)
        np.save(embeds_file, embeds_np)

        return vocab_np, embeds_np

    def make_vocab_and_embeds_glove(self, vocab_file, embeds_file):
        vocab, embeds = [], []
        with open(self.embed_dir_unprocessed, 'rt') as f:
            everything = f.read().strip().split('\n')
        for idx in range(len(everything)):
            idx_word = everything[idx].split(' ')[0]
            idx_embeds = [float(val) for val in everything[idx].split(' ')[1:]]
            vocab.append(idx_word)
            embeds.append(idx_embeds)

        vocab_np = np.array(vocab)  # vocab_size
        embeds_np = np.array(embeds)  # vocab_size X embed_dim

        with open(vocab_file, 'wb') as f:
            np.save(f, vocab_np)

        with open(embeds_file, 'wb') as f:
            np.save(f, embeds_np)

        return vocab_np, embeds_np

    def process_data(self, data, train_split=True, rewriting=False,
                     last_checkpoint_path=None, first_shard=True):
        if self.embed_type not in ['none']:
            data = self.process_data_embeds(
                data, train_split=train_split, rewriting=rewriting,
                last_checkpoint_path=last_checkpoint_path,
                first_shard=first_shard)
        else:
            data = self.process_data_no_embeds(
                data, train_split=train_split, rewriting=rewriting,
                last_checkpoint_path=last_checkpoint_path,
                first_shard=first_shard)
        return data

    def process_data_embeds(self, data, train_split=True, rewriting=False,
                            last_checkpoint_path=None, first_shard=True):
        '''
        Procedure:
            1. Tokenize and vectorize raw data
            2. Get the most frequent tokens
            3. Recreate the vocabulary, embeddings and word2idx/idx2word
               dictionaries based on only the most frequent tokens (or based
               on existing saved files when rewriting.)
            4. Reindex the data and pad
        Additionally, if prepending labels:
            Get a set of all labels in the training set as strings.
            Prepend 'LABEL_' to each string in case it is a duplicate
            vocabulary item.
            Create corresponding indexes for each string label, starting from
            the last vocabulary index (self.vocab_size + 4).
            Add both strings and indexes to self.word2idx and self.idx2word
            ...
        '''
        if self.prepend_labels and train_split and first_shard:
            labels = [doc['label'] for doc in data]
            str_labels = ['LABEL_' + str_lab
                          for str_lab in sorted(set(labels))]
            if first_shard:
                idx_labels = np.arange(len(str_labels)) + self.vocab_size + 4
            else:
                idx_labels = np.arange(len(str_labels)) + (self.vocab_size - len(str_labels))
        else:
            idx_labels = None

        data = self._tokenize_and_vectorize(data)

        if first_shard and train_split:
            if rewriting:
                try:
                    old_idx2word = copy.deepcopy(self.idx2word)
                    self.vocab, self.embeds, self.word2idx, self.idx2word =\
                        self._load_existing_compact_embeds(last_checkpoint_path)
                except:
                    print("Could not load existing word2idx and idx2word "
                          "dictionaries, rebuilding based on specified dataset. "
                          "If pre-training and rewriting on two different "
                          "datasets, MAKE SURE the vocabularies are the same "
                          "for both.")
                    top_idxs = self._get_frequency(data)
                    old_idx2word = copy.deepcopy(self.idx2word)
                    self.vocab, self.embeds, self.word2idx, self.idx2word =\
                        self._prepare_compact_embeds(top_idxs,
                                                     idx_labels=idx_labels)
            else:
                top_idxs = self._get_frequency(data)
                old_idx2word = copy.deepcopy(self.idx2word)
                self.vocab, self.embeds, self.word2idx, self.idx2word =\
                    self._prepare_compact_embeds(top_idxs,
                                                 idx_labels=idx_labels)
        else:
            old_idx2word = None

        if self.prepend_labels and train_split and first_shard:
            # Add label strings and indexes to the existing word2idx and
            # idx2word dictionaries
            for idx, lab in enumerate(str_labels):
                self.word2idx[lab] = idx_labels[idx]
                self.idx2word[idx_labels[idx]] = lab
            len_labels = len(str_labels)
        else:
            len_labels = 0

        data = self._reindex_data_and_pad(data, self.word2idx,
                                          old_idx2word=old_idx2word)

        if first_shard and train_split:
            self.vocab_size = self.vocab_size + 4 + len_labels

        return data

    def _tokenize_and_vectorize(self, dataset):
        data = []
        for doc_dict in tqdm(dataset):
            tokenized = self.tokenizer(doc_dict['text'].strip().lower())
            tensor = torch.tensor(
                [self.word2idx[token]
                 if token in self.word2idx.keys()
                 else self.word2idx[self.UNK]
                 for token in tokenized][:(self.max_seq_len-2)],
                dtype=torch.long)  # -2 for SOS+EOS tokens
            length = tensor.size()[0]
            data.append((tensor, length, doc_dict['label']))

        return data

    def _get_frequency(self, train_data):
        relevant = torch.cat([val[0] for val in train_data])
        # Ignore UNK token in counts
        relevant = relevant[relevant != self.word2idx[self.UNK]]
        counts = torch.bincount(relevant)
        top_counts, top_indexes = torch.topk(counts, self.vocab_size)
        return top_indexes

    def _prepare_compact_embeds(self, top_indexes, idx_labels=None):
        reindexes = np.arange(self.vocab_size+4)
        new_vocab = self.vocab[top_indexes]
        new_embeds = self.embeds[top_indexes, :]

        new_vocab = np.insert(new_vocab, 0, self.PAD)
        new_vocab = np.insert(new_vocab, 1, self.UNK)
        new_vocab = np.insert(new_vocab, 2, self.SOS)
        new_vocab = np.insert(new_vocab, 3, self.EOS)

        # Pad token is all 0s
        pad_emb_np = np.zeros((1, new_embeds.shape[1]))
        # UNK token is mean of all other embeds
        unk_emb_np = np.mean(new_embeds, axis=0, keepdims=True)
        # SOS token is a random vector (standard normal)
        sos_emb_np = np.random.normal(size=pad_emb_np.shape)
        # EOS token is a random vector (standard normal)
        eos_emb_np = np.random.normal(size=pad_emb_np.shape)

        new_embeds = np.vstack((pad_emb_np, unk_emb_np, sos_emb_np,
                                eos_emb_np, new_embeds))
        if self.prepend_labels:
            idx_emb_nps = np.random.normal(size=(
                len(idx_labels), pad_emb_np.shape[1]))
            new_embeds = np.vstack((new_embeds, idx_emb_nps))

        new_word2idx = {}
        new_idx2word = {}
        for idx, word in zip(reindexes, new_vocab):
            new_word2idx[word] = idx
            new_idx2word[idx] = word

        # Save vocabulary, embeddings, word2idx and idx2word in checkpoint
        # directory
        np.save(os.path.join(self.checkpoint_dir, 'vocab.npy'), new_vocab)
        np.save(os.path.join(self.checkpoint_dir, 'embeds.npy'), new_embeds)
        with open(os.path.join(self.checkpoint_dir, 'word2idx.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(new_word2idx, f, ensure_ascii=False, indent=4,
                      cls=NpEncoder)

        return new_vocab, new_embeds, new_word2idx, new_idx2word

    def _load_existing_compact_embeds(self, last_checkpoint_path):
        '''
        For rewriting mode, load the pre-trained vocabulary, embeddings,
        word2idx and idx2word dictionaries.
        '''
        checkpoint_dir = os.path.abspath(
            os.path.join(last_checkpoint_path, os.pardir))
        new_vocab = np.load(os.path.join(checkpoint_dir, 'vocab.npy'))
        new_embeds = np.load(os.path.join(checkpoint_dir, 'embeds.npy'))
        with open(os.path.join(checkpoint_dir, 'word2idx.json'), 'r',
                  encoding='utf-8') as f:
            new_word2idx = json.load(f)
        new_idx2word = {idx: word for word, idx in new_word2idx.items()}

        return new_vocab, new_embeds, new_word2idx, new_idx2word

    def _reindex_data_and_pad(self, data, compact_word2idx, old_idx2word=None):
        '''
        Carries out three tasks:
        1. Converts indexes of untrimmed vocabulary to indexes of trimmed
        vocabulary.
        2. Adds special tokens to existing tensor (SOS, EOS and PAD).
        3. Prepends the label token to the front of the sequence (after SOS
           token), if 'self.prepend_labels' is True.
        '''
        reindexed = []
        for data_point in tqdm(data):
            if old_idx2word is not None:
                old_indexes = data_point[0]
                words = [old_idx2word[idx.item()] for idx in old_indexes]
                new_indexes = torch.tensor(
                        [compact_word2idx[word]
                         if word in compact_word2idx
                         else compact_word2idx[self.UNK] for word in words],
                        dtype=torch.long)
            else:
                # If subsequent shard
                new_indexes = data_point[0]

            if self.prepend_labels:
                # In case of any new labels (generally shouldn't be the case)
                if "LABEL_" + str(data_point[2]) in self.word2idx:
                    lab = self.word2idx["LABEL_" + str(data_point[2])]
                else:
                    lab = self.word2idx[self.UNK]
                new_indexes = torch.cat(
                    (torch.tensor([lab]), new_indexes[:(self.max_seq_len-3)]),
                    dim=0)
                    # restricting to 'self.max_seq_len-3' to account for the
                    # SOS, EOS and label tokens
            else:
                lab = data_point[2]

            new_indexes = torch.cat(
                    (torch.tensor([compact_word2idx[self.SOS]]),
                     new_indexes,
                     torch.tensor([compact_word2idx[self.EOS]])), dim=0)
            new_length = data_point[1] + 2

            while new_indexes.shape[0] < self.max_seq_len:
                new_indexes = torch.cat(
                        (new_indexes,
                         torch.tensor([compact_word2idx[self.PAD]])))

            reindexed.append((new_indexes, new_length, lab))

        return reindexed

    def process_data_no_embeds(self, data, train_split=True, rewriting=False,
                               last_checkpoint_path=None, first_shard=True):
        # Only prepare the vocabulary based on the first shard
        # (dataset should be large enough, e.g. Wikipedia)
        if first_shard and train_split:
            data = self._process_data_no_embeds_first_shard(
                data, rewriting=rewriting,
                last_checkpoint_path=last_checkpoint_path)
        else:
            data = self._process_data_no_embeds_subsequent_shard(
                data, rewriting=rewriting,
                last_checkpoint_path=last_checkpoint_path)
        return data

    def _process_data_no_embeds_first_shard(
            self, data, last_checkpoint_path=None, rewriting=False):
        def encode(examples):
            examples['text'] = [self.tokenizer(doc.strip().lower())
                                for doc in examples['text']]
            return examples

        # Multiprocessing for larger datasets
        threshold = 50000
        if len(data) > threshold:
            proc_num = os.cpu_count()
        else:
            proc_num = None

        data = data.map(encode, batched=True, num_proc=proc_num)
        labels = data['label']

        if rewriting:
            # Loading the large idx2word for the full vocab
            try:
                checkpoint_dir = os.path.abspath(
                    os.path.join(last_checkpoint_path, os.pardir))
                with open(os.path.join(checkpoint_dir,
                          'large_idx2word.json'), 'r',
                          encoding='utf-8') as f:
                    self.idx2word = json.load(f)
                self.idx2word = {int(idx): token
                                 for idx, token in self.idx2word.items()}
            except FileNotFoundError:
                print("Could not load existing FULL word2idx and idx2word "
                      "dictionaries, rebuilding based on specified dataset. "
                      "If pre-training and rewriting on two different "
                      "datasets, MAKE SURE the vocabularies are the same "
                      "for both.")
                idx2word = []
                for idx, doc in tqdm(enumerate(data)):
                    idx2word += doc['text']
                    if idx % 5000 == 0:
                        idx2word = list(set(idx2word))
                idx2word = list(set(idx2word))
                self.idx2word = {idx: token
                                 for idx, token in enumerate(idx2word)}

        else:
            idx2word = []
            for idx, doc in tqdm(enumerate(data)):
                idx2word += doc['text']
                if idx % 5000 == 0:
                    idx2word = list(set(idx2word))
            idx2word = list(set(idx2word))

            self.idx2word = {idx: token for idx, token in enumerate(idx2word)}

            with open(os.path.join(self.checkpoint_dir,
                      'large_idx2word.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(self.idx2word, f, ensure_ascii=False, indent=4,
                          cls=NpEncoder)

        # Additionally adding <UNK> (will be reindexed later)
        self.idx2word[len(self.idx2word) + 1] = self.UNK

        self.word2idx = {token: idx for idx, token in self.idx2word.items()}

        data = data.map(self._encode, batched=True, num_proc=proc_num)

        data = [(torch.tensor(indexes), length, label)
                  for indexes, length, label in zip(
                      data['encoded'], data['length'], data['label'])]

        if len(self.idx2word) < self.vocab_size:
            print(f"Specified vocabulary size as {self.vocab_size}, but number of unique words in dataset {len(self.idx2word)}. Setting vocabulary size to {len(self.idx2word)-1}.")
            self.vocab_size = len(self.idx2word) - 1

        if rewriting:
            # Loading the small idx2word
            try:
                old_idx2word = copy.deepcopy(self.idx2word)
                checkpoint_dir = os.path.abspath(
                    os.path.join(last_checkpoint_path, os.pardir))
                with open(os.path.join(checkpoint_dir, 'idx2word.json'), 'r',
                          encoding='utf-8') as f:
                    self.idx2word = json.load(f)
                self.idx2word = {int(idx): token for idx, token in self.idx2word.items()}
                self.word2idx = {token: idx for idx, token in self.idx2word.items()}
            except FileNotFoundError:
                print("Could not load existing TRIMMED word2idx and idx2word "
                      "dictionaries, rebuilding based on specified dataset. "
                      "If pre-training and rewriting on two different "
                      "datasets, MAKE SURE the vocabularies are the same "
                      "for both.")
                top_indexes = self._get_frequency(data)
                old_idx2word = copy.deepcopy(self.idx2word)
                reindexes = np.arange(self.vocab_size+4)
                vocab = np.vectorize(self.idx2word.get)(top_indexes)
                vocab = np.insert(vocab, 0, self.PAD)
                vocab = np.insert(vocab, 1, self.UNK)
                vocab = np.insert(vocab, 2, self.SOS)
                vocab = np.insert(vocab, 3, self.EOS)
                new_word2idx = {}
                new_idx2word = {}
                for idx, word in zip(reindexes, vocab):
                    new_word2idx[word] = idx
                    new_idx2word[idx] = word
                self.idx2word = new_idx2word
                self.word2idx = new_word2idx

                with open(os.path.join(self.checkpoint_dir, 'idx2word.json'), 'w',
                          encoding='utf-8') as f:
                    json.dump(self.idx2word, f, ensure_ascii=False, indent=4,
                              cls=NpEncoder)

        else:
            top_indexes = self._get_frequency(data)
            old_idx2word = copy.deepcopy(self.idx2word)
            reindexes = np.arange(self.vocab_size+4)
            vocab = np.vectorize(self.idx2word.get)(top_indexes)
            vocab = np.insert(vocab, 0, self.PAD)
            vocab = np.insert(vocab, 1, self.UNK)
            vocab = np.insert(vocab, 2, self.SOS)
            vocab = np.insert(vocab, 3, self.EOS)
            new_word2idx = {}
            new_idx2word = {}
            for idx, word in zip(reindexes, vocab):
                new_word2idx[word] = int(idx)
                new_idx2word[int(idx)] = word
            self.idx2word = new_idx2word
            self.word2idx = new_word2idx

            with open(os.path.join(self.checkpoint_dir, 'idx2word.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(self.idx2word, f, ensure_ascii=False, indent=4,
                          cls=NpEncoder)

        if self.prepend_labels:
            str_labels = ['LABEL_' + str(str_lab)
                          for str_lab in sorted(set(labels))]
            idx_labels = np.arange(len(str_labels)) + len(self.idx2word)
            # Add label strings and indexes to the existing word2idx and
            # idx2word dictionaries
            for idx, lab in enumerate(str_labels):
                self.word2idx[lab] = idx_labels[idx]
                self.idx2word[idx_labels[idx]] = lab

        data = self._reindex_data_and_pad(data, self.word2idx,
                                          old_idx2word=old_idx2word)

        self.vocab_size = len(self.idx2word)

        return data

    def _process_data_no_embeds_subsequent_shard(
            self, data, last_checkpoint_path=None, rewriting=False):
        def encode(examples):
            examples['text'] = [self.tokenizer(doc.strip().lower())
                                for doc in examples['text']]
            return examples

        # Multiprocessing for larger datasets
        threshold = 50000
        if len(data) > threshold:
            proc_num = os.cpu_count()
        else:
            proc_num = None

        data = data.map(encode, batched=True, num_proc=proc_num)
        data = data.map(self._encode, batched=True)
        data = [(torch.tensor(indexes), length, label)
                for indexes, length, label in zip(
                    data['encoded'], data['length'], data['label'])]
        # Only padding and adding sos/eos tokens in this case
        data = self._reindex_data_and_pad(data, self.word2idx)

        return data

    def _encode(self, examples):
        encoded = [[self.word2idx[tok] if tok in self.word2idx else self.word2idx[self.UNK] for tok in doc] for doc in examples['text']]
        examples['encoded'] = [torch.tensor(enc_doc[:(self.max_seq_len-2)]) for enc_doc in encoded]
        examples['length'] = [doc.size()[0] for doc in examples['encoded']]
        return examples


class Preprocessor_for_Transformer(Preprocessor):
    def __init__(self, transformer_type='bert-base-uncased', **kwargs):
        super().__init__(**kwargs)

        self.transformer_type = transformer_type

        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_type)

        self.PAD = self.tokenizer.pad_token
        self.UNK = self.tokenizer.unk_token

        self.vocab, self.embeds, self.word2idx, self.idx2word = (None,)*4
        self.lab_str2int = None
        self.lab_int2str = None
        self.num_labels = None

        if self.prepend_labels:
            raise Exception(
                "Prepending labels not yet available for transformer-based "
                "models.")

    def process_data(self, data, train_split=True, first_shard=True):
        if np.count_nonzero(data['label']) == 0:
            sorted_labels = [0.0]
        else:
            sorted_labels = sorted(set(
                data[idx]['label'] for idx in range(len(data))))

        if not train_split:
            sorted_val_labels = [lab for lab in sorted_labels
                                 if lab not in self.lab_str2int]
            for val in sorted_val_labels:
                self.lab_str2int[val] = len(self.lab_str2int)
        else:
            lab_str2int = {string: idx
                           for idx, string in enumerate(sorted_labels)}
            self.lab_str2int = lab_str2int

        lab_int2str = {val: key for key, val in self.lab_str2int.items()}
        self.lab_int2str = lab_int2str
        self.num_labels = len(self.lab_str2int)

        def encode(examples):
            if None in examples['text']:
                print("***** None found *****")
                examples['text'] = [text if text is not None else ''
                                    for text in examples['text']]
            tokenized_batch = self.tokenizer(
                examples['text'], truncation=True,
                max_length=self.max_seq_len, padding='max_length')
            tokenized_batch['label'] =\
                [self.lab_str2int[lab] for lab in examples['label']]
            return tokenized_batch

        data = self.map_data_split(data, encode)

        return data

    def map_data_split(self, data_split, encode):
        data_split = data_split.map(encode, batched=True)
        data_split = data_split.map(
                lambda examples: {'labels': examples['label']}, batched=True)
        data_split = data_split.remove_columns('label')
        data_split.set_format(
            type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        if not data_split.features['labels'] == Value('int64'):
            new_features = data_split.features.copy()
            new_features['labels'] = Value('int64')
            data_split = data_split.cast(new_features)
        return data_split


class Custom_Preprocessor(Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError

    def _process_data(self, dataset):
        pass
