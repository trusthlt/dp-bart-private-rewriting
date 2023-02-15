import os
import re
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset
from copy import deepcopy
from tqdm import tqdm
import pdb
from preprocessing import Preprocessor_for_RNN, Preprocessor_for_Transformer, Custom_Preprocessor
from download import download_asset

#os.environ["HF_DATASETS_OFFLINE"] = '1'

# 'dataset' classes need to take the following hyperparameters:

# Required:
# dataset_name, in_dir, out_data_dir, data_dir,
# max_seq_len, batch_size, private, local, mod_type,
# mode (might not need, just use subclasses)

# Optional:
# vec_dir, vocab_size, embed_size (RNN-based),
# transformer_type (transformer-based), epsilon (private),
# privatized_validation (downstream mode)


class DPRewriteDataset(object):
    def __init__(self, dataset_name, data_dir, checkpoint_dir, max_seq_len,
                 batch_size, mode='pretrain', train_ratio=0.9,
                 embed_type='glove', embed_size=300, embed_dir_processed=None,
                 embed_dir_unprocessed=None, vocab_size=None,
                 model_type='transformer', private=False,
                 prepend_labels=False, transformer_type='bert-base-uncased',
                 length_threshold=None, custom_preprocessor=False,
                 data_split_cutoff=None,
                 local=False, last_checkpoint_path=False,
                 custom_train_path=None, custom_valid_path=None,
                 custom_test_path=None, downstream_test_data=None):
        self.dataset_name = dataset_name

        # main directory where data is stored (all modes; e.g. imdb, yelp)
        self.data_dir = data_dir

        self.checkpoint_dir = checkpoint_dir
        self.last_checkpoint_path = last_checkpoint_path

        # vocabulary and embeddings directory (renamed from 'in_dir')
        # used after processing vectors from below 'vec_model_dir'
        self.embed_dir_processed = embed_dir_processed

        # downloaded pre-trained embedding model directory
        self.embed_dir_unprocessed = embed_dir_unprocessed

        self.mode = mode

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.model_type = model_type
        self.embed_type = embed_type
        self.transformer_type = transformer_type
        self.prepend_labels = prepend_labels
        self.length_threshold = length_threshold
        self.train_ratio = train_ratio
        self.data_split_cutoff = data_split_cutoff

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.sample_size = None
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

        self.custom_train_path = custom_train_path
        self.custom_valid_path = custom_valid_path
        self.custom_test_path = custom_test_path

        self.downstream_test_data = downstream_test_data

        if model_type == 'transformer' and not custom_preprocessor:
            self.preprocessor = Preprocessor_for_Transformer(
                    checkpoint_dir=checkpoint_dir,
                    transformer_type=transformer_type,
                    max_seq_len=max_seq_len, batch_size=batch_size,
                    prepend_labels=prepend_labels, mode=mode)
        elif model_type == 'rnn' and not custom_preprocessor:
            if embed_dir_processed is None:
                raise Exception("Please specify 'embed_dir_processed' for RNN-based models.")
            self.preprocessor = Preprocessor_for_RNN(
                    embed_dir_processed, embed_dir_unprocessed,
                    vocab_size=vocab_size, embed_type=embed_type,
                    embed_size=embed_size, checkpoint_dir=checkpoint_dir,
                    max_seq_len=max_seq_len, batch_size=batch_size,
                    prepend_labels=prepend_labels, mode=mode)
        else:
            print("Using custom preprocessor...")
            self.preprocessor = Custom_Preprocessor()

        self.private = private
        self.local = local
        if local or mode == 'rewrite':
            self.shuffle = False
        else:
            self.shuffle = True

    def load_and_process(self):
        self.load()
        self.process()
        self.prepare_dataloader()

    def load(self, subset=None):
        '''
        Description
        -----------
        Prepares a 'Dataset' object, with features consisting of 'text' and
        'label'.

        Parameters
        ----------
        subset : ``int``, Don't load the full dataset, only up to a certain
                 index.
        '''
        if self.dataset_name == 'imdb':
            print("Preparing IMDb dataset...")
            self._load_hf('imdb')
        elif self.dataset_name == 'atis':
            print("Preparing ATIS dataset...")
            self._load_from_path(valid=True, test=True)
        elif self.dataset_name == 'snips_2016':
            print("Preparing SNIPS dataset (2016 version)...")
            self._load_hf('snips_built_in_intents')
        elif self.dataset_name == 'snips_2017':
            print("Preparing SNIPS dataset (2017 version)...")
            self._load_from_path(valid=True, test=True)
        elif self.dataset_name == 'drugscom_reviews_rating':
            print("Preparing Drugs.com reviews dataset (ratings as labels)...")
            self._load_from_path(
                valid=False, test=True,
                prepare_script=prepare_drugscom_dataset,
                asset_dir=self.data_dir, predict_rating=True)
        elif self.dataset_name == 'drugscom_reviews_condition':
            print("Preparing Drugs.com reviews dataset "
                  "(conditions as labels)...")
            self._load_from_path(
                valid=False, test=True,
                prepare_script=prepare_drugscom_dataset,
                asset_dir=self.data_dir, predict_rating=False)
        elif self.dataset_name == 'reddit_mental_health':
            print("Preparing Reddit mental health dataset...")
            self._load_from_path(
                valid=False, test=False,
                prepare_script=prepare_reddit_mental_health_dataset,
                asset_dir=self.data_dir)
        elif self.dataset_name == 'amazon_reviews_books':
            print("Preparing Amazon reviews dataset ('Books_v1_00' subset)...")
            target_column_dict = {"star_rating": "label",
                                  "review_body": "text"}

            subset = 'Books_v1_00'
            self._load_hf('amazon_us_reviews', subset=subset,
                          target_column_dict=target_column_dict)
        elif self.dataset_name == 'amazon_reviews_subset':
            print("Preparing Amazon reviews dataset (framework subset)...")
            self._load_from_path(
                valid=False, test=True,
                prepare_script=prepare_amazon_subset,
                asset_dir=self.data_dir)

        elif self.dataset_name == 'openwebtext':
            print("Preparing Openwebtext dataset...")
            self._load_hf('openwebtext')
        elif self.dataset_name == 'wikipedia':
            print("Preparing Wikipedia dataset...")
            self._load_hf('wikipedia', subset='20200501.en')
        else:
            print("Preparing custom dataset...")
            self._load_custom()

    def process(self):
        '''
        Description
        -----------
        Applies `_process_split()` method to each data split.
        '''
        self.train_data = self._process_split(self.train_data,
                                              train_split=True)
        if self.valid_data is not None:
            self.valid_data = self._process_split(self.valid_data)
        if self.test_data is not None:
            self.test_data = self._process_split(self.test_data)

    def prepare_dataloader(self):
        self.train_iterator = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=self.shuffle)
        self.sample_size = len(self.train_data)
        print('Num training:', self.sample_size)

        if self.valid_data is not None:
            self.valid_iterator = DataLoader(
                self.valid_data, batch_size=self.batch_size,
                shuffle=self.shuffle)
            print('Num validation:', len(self.valid_data))

        if self.test_data is not None:
            self.test_iterator = DataLoader(
                self.test_data, batch_size=self.batch_size,
                shuffle=self.shuffle)
            print('Num test:', len(self.test_data))

    def _load_hf(self, name, subset=None, target_column_dict=None,
                 large=False):
        '''
        Description
        -----------
        Loads a dataset from huggingface

        Parameters
        ----------
        name : ``str``, Specific name of a dataset as it is called in HF
               datasets
               E.g. 'wikipedia'
        subset : ``str``, Subset of a dataset as it is called in HF datasets
                 E.g. '20200501.en'
        split_name : ``str``, name of a particular data split as it is called
                     in HF datasets
                     E.g. 'train'
        target_column_dict : ``dict``, If a HF dataset does not have only
                             'text' and 'label' columns, a dictionary can be
                             provided that specifies which column names should
                             be considered as 'text' and 'label'.
                             E.g. {'star_rating': 'label',
                                   'review_body': 'text'}
                                  (Amazon reviews dataset)
        large : ``bool``, Whether the dataset to be loaded is large or not
                (in this case it's split up into multiple 'shards').
        '''
        # Preparing the cache dir
        cache_dir = os.path.join(self.data_dir, self.dataset_name)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if subset is not None:
            data = load_dataset(name, subset, cache_dir=cache_dir)
        else:
            data = load_dataset(name, cache_dir=cache_dir)

        # If specific column names specified as 'text' and 'label'
        # (all others are discarded)
        if target_column_dict is not None:
            for col_orig, col_target in target_column_dict.items():
                data = data.rename_column(col_orig, col_target)
            # Assuming column names are the same for different splits
            # (based on train split columns)
            data = data.remove_columns(
                [col for col in data.column_names['train']
                 if col not in ['text', 'label']])

        # Adding placeholder labels in case dataset does not have its own
        for split in data.keys():
            if 'label' not in data[split].column_names:
                data[split] = data[split].add_column("label", np.zeros(len(data[split])))
        # Applying the data split cut-off if specified
        if self.data_split_cutoff is not None:
            data['train'] = data['train'].select(
                list(range(self.data_split_cutoff)))

        # Preparing the validation split
        if 'validation' not in data and self.mode != 'rewrite':
            data_split = data['train'].train_test_split(test_size=(1-self.train_ratio))

            self.train_data = data_split['train']
            self.valid_data = data_split['test']
        elif 'validation' in data:
            self.train_data = data['train']
            self.valid_data = data['validation']
        else:
            self.train_data = data['train']

        # Preparing the test split, if available
        if 'test' in data and self.mode != 'pretrain':
            self.test_data = data['test']

    def _load_from_path(self, valid=True, test=True, prepare_script=None,
                        **kwargs):
        try:
            train_csv_path = os.path.join(self.data_dir, self.dataset_name,
                                          f'{self.dataset_name}_train.csv')
            train_data = load_dataset("csv", data_files=train_csv_path,
                                      column_names=["label", "text"])
        except FileNotFoundError:
            print(f"Could not find processed dataset, preparing dataset from "
                  f"path: {self.data_dir} (download and save raw files here).")
            download_asset(self.dataset_name, asset_dir=self.data_dir)
            if prepare_script is not None:
                print(f"Processing downloaded raw files for "
                      f"{self.dataset_name} dataset...")
                prepare_script(kwargs['asset_dir'])
            else:
                raise Exception(f"No script provided for preparing dataset "
                                f"files from raw data (dataset: "
                                f"{self.dataset_name}).")
            train_csv_path = os.path.join(
                    self.data_dir, self.dataset_name,
                    f'{self.dataset_name}_train.csv')
            train_data = load_dataset("csv", data_files=train_csv_path,
                                      column_names=["label", "text"])

        # Applying the data split cut-off if specified
        if self.data_split_cutoff is not None:
            train_data['train'] = train_data['train'].select(
                list(range(self.data_split_cutoff)))

        if valid:
            valid_csv_path = os.path.join(
                self.data_dir, self.dataset_name,
                f'{self.dataset_name}_valid.csv')
            valid_data = load_dataset("csv", data_files=valid_csv_path,
                                      column_names=["label", "text"])

            self.train_data = train_data['train']
            self.valid_data = valid_data['train']
        elif not valid and self.mode != 'rewrite':
            data_split = train_data['train'].train_test_split(
                test_size=(1-self.train_ratio))

            self.train_data = data_split['train']
            self.valid_data = data_split['test']
        else:
            self.train_data = train_data['train']

        if test and self.mode != 'pretrain':
            test_csv_path = os.path.join(
                self.data_dir, self.dataset_name,
                f'{self.dataset_name}_test.csv')
            test_data = load_dataset("csv", data_files=test_csv_path,
                                     column_names=["label", "text"])
            self.test_data = test_data['train']

    def _load_custom(self):
        if self.custom_train_path is not None:
            self.train_data = self._load_custom_split(self.custom_train_path)
        else:
            raise Exception(
                f"{self.dataset_name} not in currently prepared datasets, "
                f"but 'custom_train_path' is None. Please either specify a "
                f"dataset name among existing datasets, or add a custom "
                f"dataset path.")

        # Applying the data split cut-off if specified
        if self.data_split_cutoff is not None:
            self.train_data = self.train_data.select(
                list(range(self.data_split_cutoff)))

        if self.custom_valid_path is not None and \
           self.custom_valid_path.lower() != 'none':
            self.valid_data = self._load_custom_split(self.custom_valid_path)
        else:
            # If no validation path specified, make a split from the
            # training set
            data_split = self.train_data.train_test_split(
                test_size=(1-self.train_ratio))
            self.train_data = data_split['train']
            self.valid_data = data_split['test']

        if self.custom_test_path is not None and \
           self.custom_test_path.lower() != 'none':
            self.test_data = self._load_custom_split(self.custom_test_path)

        if self.mode == 'downstream' and \
                self.downstream_test_data is not None and \
                self.downstream_test_data.lower() != 'none':
            print(f"Loading original test set for dataset {self.downstream_test_data}...")
            self.test_data = self._load_downstream_test_set(self.downstream_test_data)

    def _load_custom_split(self, path):
        data = load_dataset('csv', data_files=path,
                            column_names=["label", "text"])
        data = data['train']
        if np.all(np.array(data['text']) == None):
            # If there is only one column in the CSV file, then the
            # second column in the dataset will only have None, hence
            # need to remove it and rename the first column
            data = data.remove_columns("text")
            data = data.rename_column("label", "text")
            data = data.add_column("label", np.zeros(len(data)))
            if self.prepend_labels:
                raise Exception(
                    "Requested option to prepend labels to each dataset "
                    "tensor, but provided CSV file has no labels.")
        return data

    def _process_split(self, data, train_split=False):
        '''
        Description
        -----------
        Carries out preprocessing on the loaded dataset (additional sharding
        process for larger datasets).
        Resulting preprocessed dataset:
            len(data): length of dataset split
            data[i][0]: torch tensor of max_seq_len
            data[i][1]: length of tensor
            data[i][2]: label string

        Parameters
        ----------
        data : ``Dataset``, Loaded dataset object.
        '''
        # Optionally removing parts of the dataset, where the token count is
        # lower than a given threshold (based on whitespace split)
        if self.length_threshold is not None and \
                str(self.length_threshold).lower() != 'none':
            data = data.filter(
                lambda example: len(
                    example['text'].split()) <= self.length_threshold if example['text'] is not None else False)

        threshold = 2000000
        if len(data) > threshold:
            # Preprocessing for large datasets
            num_shards = 4
            new_shards = []
            print(f"Dataset very large, splitting preprocessing into "
                  f"{num_shards} shards.")
            for idx in range(num_shards):
                if idx > 0:
                    first_shard = False
                else:
                    first_shard = True
                new_shard = self.preprocessor.process_data(
                    data.shard(num_shards=num_shards, index=idx),
                    first_shard=first_shard, train_split=train_split)
                new_shards += new_shard
            data = new_shards
        else:
            data = self.preprocessor.process_data(
                data, train_split=train_split, first_shard=True)
        return data

    def _load_downstream_test_set(self, name):
        cache_dir = os.path.join(self.data_dir, name)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # So far only IMDb has a test set out of these
        hf_datasets = ['imdb', 'snips_2016', 'amazon_reviews_books']

        if name in hf_datasets:
            if name != 'imdb':
                raise Exception(f"{name} does not have a test set.")
            data = load_dataset(name, split='test', cache_dir=cache_dir)
        else:
            test_csv_path = os.path.join(
                self.data_dir, name,
                f'{name}_test.csv')
            test_data = load_dataset("csv", data_files=test_csv_path,
                                     column_names=["label", "text"])
            data = test_data['train']

        return data


def prepare_drugscom_dataset(asset_dir=None, predict_rating=True):
    rat_str = 'rating' if predict_rating else 'condition'
    train_data_path = os.path.join(asset_dir, 'raw',
                                   f'drugscom_reviews_{rat_str}',
                                   'drugsComTrain_raw.tsv')
    test_data_path = os.path.join(asset_dir, 'raw',
                                  f'drugscom_reviews_{rat_str}',
                                  'drugsComTest_raw.tsv')
    paths = [train_data_path, test_data_path]
    for path in paths:
        split = 'train' if 'Train' in path else 'test'

        df = pd.read_csv(path, sep='\t')
        df = df.drop(columns=['Unnamed: 0', 'drugName', 'date', 'usefulCount'])

        if predict_rating:
            df = df.drop(columns=['condition'])
            df = df[df.columns[::-1]]
            # Converting ratings to 2 classes, as in Shiju and He 2021
            df['rating'].loc[df['rating'] < 8] = 0
            df['rating'].loc[df['rating'] >= 8] = 1

            # Removing start and end double-quotes
            df['review'] = df['review'].str.slice(start=1, stop=-1)

            name = 'rating'
        else:
            df = df.drop(columns=['rating'])
            name = 'condition'

        mid_path = os.path.join(asset_dir, f'drugscom_reviews_{rat_str}')
        if not os.path.exists(mid_path):
            os.makedirs(mid_path)
        out_path = os.path.join(mid_path,
                                f'drugscom_reviews_{name}_{split}.csv')
        df.to_csv(out_path, header=False, index=False)


def prepare_reddit_mental_health_dataset(asset_dir=None):
    data_path = os.path.join(asset_dir, 'raw', 'reddit_mental_health')
    files = os.listdir(data_path)
    full_df = []
    for f in tqdm(files):
        file_path = os.path.join(data_path, f)
        df = pd.read_csv(file_path)
        df = df.drop(df.columns.difference(['subreddit', 'post']), axis=1)
        full_df.append(df)
    full_df = pd.concat(full_df, ignore_index=True)
    full_df = full_df.drop_duplicates(keep='first', ignore_index=True)

    mid_path = os.path.join(asset_dir, 'reddit_mental_health')
    if not os.path.exists(mid_path):
        os.makedirs(mid_path)
    out_path = os.path.join(mid_path, 'reddit_mental_health_train.csv')
    full_df.to_csv(out_path, header=False, index=False)


def prepare_amazon_subset(asset_dir=None):
    '''
    Prepares the framework's Amazon reviews subset from train and test indices
    that come with the framework (prepared from
    `prepare_amazon_subset_initial`).
    '''
    target_column_dict = {"star_rating": "label",
                          "review_body": "text"}
    subsets = ['Digital_Video_Games_v1_00', 'Electronics_v1_00',
               'Lawn_and_Garden_v1_00', 'Major_Appliances_v1_00',
               'Mobile_Apps_v1_00', 'Office_Products_v1_00', 'Wireless_v1_00']
    loaded_subsets_train = []
    loaded_subsets_test = []

    train_indices_path = os.path.join(
        asset_dir, 'amazon_reviews_subset',
        'amazon_reviews_subset_train_indices.csv')
    test_indices_path = os.path.join(asset_dir, 'amazon_reviews_subset',
                                     'amazon_reviews_subset_test_indices.csv')
    train_indices = pd.read_csv(train_indices_path, index_col=None)
    test_indices = pd.read_csv(test_indices_path, index_col=None)

    for subset in tqdm(subsets):
        # Step 1: Load dataset subsets from HF
        cache_dir = os.path.join(asset_dir, 'amazon_reviews_books')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        data = load_dataset('amazon_us_reviews', subset, cache_dir=cache_dir)

        for col_orig, col_target in target_column_dict.items():
            data = data.rename_column(col_orig, col_target)
        data = data.remove_columns(
            [col for col in data.column_names['train']
             if col not in ['text', 'label']])
        data = data['train']

        # Step 2: Binarize labels (4 and 5: 1)
        data = data.map(
            lambda example: {'label': 1 if example['label'] in [4, 5] else 0})

        # Step 3: Select data from loaded indexes
        train_indices_subset =\
            train_indices[train_indices["subset"] == subset]["original_idx"].tolist()
        test_indices_subset =\
            test_indices[test_indices["subset"] == subset]["original_idx"].tolist()

        train_data = data.select(indices=train_indices_subset)
        test_data = data.select(indices=test_indices_subset)

        loaded_subsets_train.append(train_data)
        loaded_subsets_test.append(test_data)

    train_data = concatenate_datasets(loaded_subsets_train)
    test_data = concatenate_datasets(loaded_subsets_test)

    # Save to CSV
    out_dir = os.path.join(asset_dir, 'amazon_reviews_subset')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_path = os.path.join(out_dir, 'amazon_reviews_subset_train.csv')
    test_path = os.path.join(out_dir, 'amazon_reviews_subset_test.csv')

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)


def prepare_amazon_subset_initial(asset_dir=None, binary_5_only=False,
                                  balance_labels=True, length_threshold=20,
                                  reduce_n_mod=3):
    '''
    Description
    -----------
    Prepares the framework's Amazon reviews subset for the first
    time (getting the train-test indices).

    Parameters
    ----------
    asset_dir : `str`, General directory where to find and store all datasets.
    binary_5_only : `bool`, If True, prepare binary classification where
                    [5] --> 1, [1,2,3,4] --> 0.
                    If False, prepare binary classification where
                    [4,5] --> 1, [1,2,3] --> 0
    balance_labels : `bool`, If True, make positive and negative label count
                     equal (reducing the larger class).
    length_threshold : `int`, Maximum document length based on number of
                       tokens, where the document is split by whitespace.
    reduce_n_mod : `int`, Whether to reduce the size of the dataset further,
                   by 1/nth of its size. E.g. If `reduce_n_mod == 3`, then
                   the dataset is reduced by 1/3rd its size.
    '''
    target_column_dict = {"star_rating": "label",
                          "review_body": "text"}
    subsets = ['Digital_Video_Games_v1_00', 'Electronics_v1_00',
               'Lawn_and_Garden_v1_00', 'Major_Appliances_v1_00',
               'Mobile_Apps_v1_00', 'Office_Products_v1_00', 'Wireless_v1_00']
    loaded_subsets_train = []
    loaded_subsets_test = []
    for subset in tqdm(subsets):
        # Step 1: Load dataset subsets from HF
        cache_dir = os.path.join(asset_dir, 'amazon_reviews_books')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        data = load_dataset('amazon_us_reviews', subset, cache_dir=cache_dir)

        for col_orig, col_target in target_column_dict.items():
            data = data.rename_column(col_orig, col_target)
        data = data.remove_columns(
            [col for col in data.column_names['train']
             if col not in ['text', 'label']])
        data = data['train']

        # Adding original indexes and subset info for reproducibility
        data = data.map(lambda example, idx: {'original_idx': idx, 'subset': subset}, with_indices=True)

        # Step 2: Filter documents with number of tokens above LT
        #         and remove empty strings
        if length_threshold is not None:
            data = data.filter(lambda example: len(
                    example['text'].split()) <= length_threshold)

        # Step 3: Binarize labels
        if binary_5_only:
            data = data.map(lambda example: {'label': 1 if example['label'] == 5 else 0})
            name_appendix = 'binary5'
        else:
            data = data.map(lambda example: {'label': 1 if example['label'] in [4, 5] else 0})
            name_appendix = 'binary45'

        if balance_labels:
            ones = data.filter(lambda example: example['label'] == 1)
            zeros = data.filter(lambda example: example['label'] == 0)
            np.random.seed(seed=0)
            # From an array arange(len(ones)), select len(zeros) values
            chosen_indices = np.random.choice(len(ones), len(zeros))
            ones_kept = ones.select(chosen_indices)
            data = concatenate_datasets([ones_kept, zeros])

        if reduce_n_mod is not None:
            data = data.filter(lambda example, idx: idx % reduce_n_mod != 0,
                               with_indices=True)

        # Step 4: Split train-test splits (stratified for the different classes)
        train_ratio = 0.9
        classes, y_indices = np.unique(data.with_format("numpy")['label'],
                                       return_inverse=True)
        n_classes = classes.shape[0]
        class_counts = np.bincount(y_indices)
        class_indices = np.split(np.argsort(y_indices, kind="mergesort"),
                                 np.cumsum(class_counts)[:-1])
        train = []
        test = []
        for cl in range(n_classes):
            num_train = int(class_counts[cl] * train_ratio)
            train_indices = class_indices[cl][:num_train]
            test_indices = class_indices[cl][num_train:]
            train.extend(train_indices)
            test.extend(test_indices)

        train_data = data.select(indices=train)
        test_data = data.select(indices=test)
        loaded_subsets_train.append(train_data)
        loaded_subsets_test.append(test_data)

    train_data = concatenate_datasets(loaded_subsets_train)
    test_data = concatenate_datasets(loaded_subsets_test)

    train_data = train_data.filter(lambda example: example['text'] != '')
    test_data = test_data.filter(lambda example: example['text'] != '')

    train_indices = train_data.remove_columns(["label", "text"])
    test_indices = test_data.remove_columns(["label", "text"])

    train_data_out = train_data.remove_columns(["original_idx", "subset"])
    test_data_out = test_data.remove_columns(["original_idx", "subset"])

    # Save to CSV
    out_dir = os.path.join(asset_dir, 'amazon_reviews_subset')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_path = os.path.join(out_dir, 'amazon_reviews_subset_train.csv')
    test_path = os.path.join(out_dir, 'amazon_reviews_subset_test.csv')
    train_indices_path = os.path.join(
        out_dir, 'amazon_reviews_subset_train_indices.csv')
    test_indices_path = os.path.join(
        out_dir, 'amazon_reviews_subset_test_indices.csv')

    train_data_out.to_csv(train_path, index=False)
    test_data_out.to_csv(test_path, index=False)
    train_indices.to_csv(train_indices_path, index=False)
    test_indices.to_csv(test_indices_path, index=False)
