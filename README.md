# Code accompanying paper "DP-BART for Privatized Text Rewriting under Local Differential Privacy"

## Description

This is an extension of the framework provided at: https://github.com/trusthlt/dp-rewrite (including rest of the README below).

## Installation

```bash
$ sudo apt-get install python3-dev
```

```bash
$ pip install -r requirements.txt
```

## Quick Start: Running Experiments

Steps to running an experiment:
1. Choose a **mode** (`pretrain`, `rewrite`, or `downstream`, see **Modes** below for more details).
2. Choose a **model**. Currently only `adept` is available for pre-training and rewriting, and `bert_downstream` for downstream classification.
3. Choose a desired dataset (see **Custom Datasets** below if you want to use your own dataset).
4. Specify a **name** for your experiment with `--name`. This will be the folder name associated with your experiment where checkpoints, configuration info, learning curves, various statistics and rewritten data will be saved to.
5. Specify two main directories: (1) `--output_dir` where the above folder will be saved and (2) `asset_dir` where datasets will be loaded from. See **Directories** below for more information.
6. Finally, you can configure additional hyperparameters for the experiment such as `--batch_size` and `--learning_rate`. For a complete list of available arguments, you can run `python main.py -h`.

For sample experiment runs, please look at the shell scripts provided in `sample_scripts`. These contain test commands covering the most important basic arguments.

Additionally, if you are on a machine with low resources, you can specify `--local True`, setting `--local_iter_size` to the desired number of iterations per epoch to run the model.

Please read the short introduction about the three available modes below.

### Modes

There are 3 modes of experiments:

1. `pretrain`: This is used for training/fine-tuning an autoencoder model on some data.
2. `rewrite`: This is used for loading a checkpoint of an autoencoder model (provide the checkpoint with param `last_checkpoint_path`) and running it over a whole data set (= rewriting the dataset).
3. `downstream`: This is used for training and/or evaluating a downstream model on some data. Can be done on the raw datasets we provide (see below) or on rewritten data (use parameters `--dataset custom` and provide the associated path with `--custom_train_path`, as well as optionally `--custom_valid_path` and `--custom_test_path`). More information in **Custom Datasets** below.

### Pre-defined Experiments

Optionally, it is possible to specify a pre-defined experiment with `--experiment` which simplifies the above **Quick Start** process by pre-selecting parameters associated with a given experiment (e.g. *mode*, *model*, etc.).

If using a pre-defined experiment, any conflicting parameters specified will be overridden.

The specific list of parameters can be seen in `prepare_specific_experiment` of `utils.py`. The currently available experiments are outlined below:

| Experiment name | Description |
| --- | --- |
| `adept_l2norm_pretrain` | Our implementation of the original pre-training procedure used in [ADePT](https://arxiv.org/abs/2102.01502) |
| `adept_l1norm_pretrain` | Modified ADePT pre-training procedure based on https://arxiv.org/abs/2109.03175 |
| `adept_l2norm_rewrite` | Our implementation of the original rewriting procedure used in [ADePT](https://arxiv.org/abs/2102.01502) |
| `adept_l1norm_rewrite` | Modified ADePT rewriting procedure from https://arxiv.org/abs/2109.03175 |

## Available Datasets

All available datasets will be downloaded and preprocessed automatically to fit the required CVS format.

| Dataset Name | Description | Reference |
|---|---|---|
| `imdb` | Dataset commonly used for binary sentiment classification. Split: 25,000 train, 25,000 test examples. | Original source: https://ai.stanford.edu/~amaas/data/sentiment/ <br>Fetched from: https://huggingface.co/datasets/imdb<br>Paper citation: [Maas et al., 2011](https://aclanthology.org/P11-1015.pdf) |
| `atis` | Airline Travel Information Systems (ATIS) dataset, originally consisting of audio recordings and manual transcripts related to flight reservations, meant for the tasks of intent detection and slot filling. Here we include the manual transcripts and intent labels of the dataset. Split: 4,478 train, 500 development and 893 test examples. | Original source: https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data<br>Paper citation: [Dahl et al., 1994](https://aclanthology.org/H94-1010.pdf) |
| `snips_2016` | Dataset collected for the Snips personal assistant, used for an NLU benchmark from 2016. No pre-defined dataset split, 328 examples in total. | Original source: https://github.com/sonos/nlu-benchmark/tree/master/2016-12-built-in-intents<br>Fetched from: https://huggingface.co/datasets/snips_built_in_intents |
| `snips_2017` | Dataset collected from the Snips personal voice assistant, used for an NLU benchmark from 2017. Split: 13,084 train, 700 development and 700 test examples. | Original source: https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines<br>Paper citation: [Coucke et al., 2018](https://arxiv.org/abs/1805.10190) |
| `drugscom_reviews_rating` | Original dataset consists of patient reviews on specific drugs and patients' related conditions, along with a rating out of 10 stars on overall satisfaction. Processed here to only include 'review' and 'rating' columns from the original dataset. Split: 161,297 train, 53,766 test examples. | Source: https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29<br>Paper citation: [Gräßer et al. 2018](https://dl.acm.org/doi/10.1145/3194658.3194677) |
| `drugscom_reviews_condition` | Original dataset consists of patient reviews on specific drugs and patients' related conditions, along with a rating out of 10 stars on overall satisfaction. Processed here to only include 'review' and 'condition' columns from the original dataset. Split: 161,297 train, 53,766 test examples. | Source: https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29<br>Paper citation: [Gräßer et al. 2018](https://dl.acm.org/doi/10.1145/3194658.3194677) |
| `reddit_mental_health` | Dataset consisting of posts from 28 subreddits from 2018-2020, with 15 of the subreddits being from mental health support groups. Processed here to only include 'subreddit' and 'post' columns from the original dataset. Split: 6,467,760 train. | Source: https://zenodo.org/record/3941387<br>Paper citation: [Low et al. 2020](https://www.jmir.org/2020/10/e22635/) |
| `amazon_reviews_books` | Large dataset consisting of customer reviews on products from Amazon.com from 1995 until 2015. Here the subset 'Books_v1_00' is used, consisting of reviews in the Books product category. Processed here to only include 'star_rating' and 'review_body' columns. Split: 10,319,090 train. | Original source: https://s3.amazonaws.com/amazon-reviews-pds/readme.html <br>Fetched from: https://huggingface.co/datasets/amazon_us_reviews |
| `wikipedia` | Currently using the Wikipedia dump from May 1, 2020. | Fetched from: https://huggingface.co/datasets/wikipedia |
| `openwebtext` | Dataset created as an open-source version of WebText from OpenAI. | Original source: https://skylion007.github.io/OpenWebTextCorpus/ <br>Fetched from: https://huggingface.co/datasets/openwebtext |

### Custom Datasets

If you want to use your **own dataset**:
- Please specify `--dataset custom`, along with  `--custom_train_path` (also optionally `--custom_valid_path` and `--custom_test_path`).
- If `--custom_valid_path` is not provided, training and validation sets will be prepared from the training path, with split ratio according to `--train_split_ratio`.
- Datasets should be in CSV format, with one column for 'text', and optionally a second column for 'labels'. If `--prepend_labels` is True, labels will be prepended to each row's text for `pretrain` and `rewrite` modes.
- Test data is not used for the `pretrain` mode.

## Available Models

### Autoencoders

| Model Name | Description |
|---|---|
| `adept` | Our reference implementation of Adept. As the Adept paper lacks info about the exact network structure, some architecture decisions were made by us. |

### Downstream Models

| Model Name | Description |
|---|---|
| `bert_downstream` | A BERT model with a feedforward head (output dimension based on the number of classes in the dataset). The specific type of transformer can be specified with `--transformer_type`. |

### Custom Models

It is also possible to include a custom model for the experiments. The procedure is as follows:

- Depending on whether the model is RNN- or Transformer-based, fill out the `CustomModel_RNN` or `CustomModel_Transformer` classes, respectively. These can be found in `models/autoencoders/custom.py`.
- The expected inputs/outputs of the `forward` method are specified in the docstrings of these custom model classes.
  - `CustomModel_RNN`:
    - Input: Dictionary of input values with key-value pairs as follows:
      - `input_ids`: `torch.LongTensor` (`batch_size` X `seq_len`)
      - `lengths`: `torch.LongTensor` (`batch_size`)
      - `teacher_forcing_ratio`: `float`
    - Return:
      - `torch.DoubleTensor` (`batch_size` X `seq_len-1` X `vocab_size+4`)
  - `CustomModel_Transformer`: 
    - Input: Dictionary of input values with key-value pairs as follows:
      - `input_ids`: `torch.LongTensor` (`batch_size` X `seq_len`)
      - `attention_mask`: `torch.LongTensor` (`batch_size` X `seq_len`)
    - Return:
      - `torch.DoubleTensor` (`batch_size` X `seq_len-1` X `vocab_size`)
- Any new arguments required for the model that are not currently part of the framework can be added with the CMD argument `--custom_model_arguments`, which will be available as a list under the custom model class `self.config.custom_config_list` variable.
  - E.g. `--custom_model_arguments alpha beta gamma` --> `self.config.custom_config_list == ['alpha', 'beta', 'gamma']`
- Other CMD arguments part of the framework can be specified as usual (e.g. `--epsilon`, `--batch_size`, etc.)

## Project Structure

- ``assets`` sub-folders contain the (raw) data of training datasets and embeddings.
- ``models``
    - ``autoencoders`` contains all (PyTorch) autoencoder models that are used to privatize the data, each in its own python file.
    - ``downstream_models`` contains all (PyTorch) models that tackle a downstream task, each in its own python file. They can be used with original or rewritten data.
- ``sample_scripts`` contains tests for different datasets and experiments.
- ``download.py`` can be used to download datasets and embeddings to the assets folder **(currently not used)**.
- ``dataload.py`` is the main dataloading script which prepares existing and custom datasets.
- ``experiments.py`` contains three types of "experiments" and the main training/evaluation loops for the framework.
- ``main.py`` can be executed from the command line with arguments to run experiments (see Quick Start above).
- ``preprocessing.py`` contains preprocessing pipelines that transform the data according to a model's needs. Separate pipelines are implemented for transformer and RNN-based models.
- ``settings.py`` specifies all the arguments that `main.py` can be run with.
- ``utils.py`` contains additional helper functions and classes (e.g. `EarlyStopping`)

## Directories

### Overview

If you want to use the **default folders** in the project structure for data, logs, stats, and so on, you don't need to
specify any paths for using autoencoders and you only need to specify `embed_dir_unprocessed` if you want to train/use
an RNN-based model with pre-trained embeddings (`--embed_type` as `glove` or `word2vec`).

If you want to **load the project in an IDE**, it's a good idea to set `output_dir` & `asset_dir` to paths outside the project folders, as some IDEs will take very long to index large amounts of data that may be written into these folders.

### Details

**Output directories**:

- `output_dir`: Stats & logs (defaults to `<project_root>/results`)
- `dump_dir`: Directory for things that might need much storage (e.g., model checkpoints & rewritten data) (defaults to `output_dir`)
- In each of these directories, data will be written to a custom experiment name (if none is given, this is set to the current timestamp)

**Input directories**:

- `asset_dir`: Where to look for assets like datasets and processed embeddings (defaults to `<project_root>/assets`)
- `embed_dir_unprocessed`: Where to look for pre-trained word embeddings when using RNN-based models.
  - Specify the actual path of the embeddings, not the parent folder.
  - 'Unprocessed' refers to the original word embedding models, not modified for specific experiments (e.g. subsampled vocabularies based on frequency in a given dataset). The 'processed' embedding directory is automatically created under `asset_dir` during preprocessing.
- `data_dir`: If you store your data somewhere else than the other assets, you can specify the path here (defaults to `asset_dir`)

All paths should be specified as global paths.
