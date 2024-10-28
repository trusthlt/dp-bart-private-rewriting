# DP-BART for Privatized Text Rewriting under Local Differential Privacy

## Description

Code accompanying "DP-BART for Privatized Text Rewriting under Local Differential Privacy" paper ( https://arxiv.org/abs/2302.07636 ).

## Installation

```bash
$ sudo apt-get install python3-dev
```

```bash
$ pip install -r requirements.txt
```

## Running Experiments

Experiments can be run using the scripts provided in `sample_scripts`. Examples are provided for the `atis` dataset at a privacy budget of $\varepsilon = 250$, but can be extended to other datasets and $\varepsilon$ values by changing the `--dataset` and `--epsilon` arguments, respectively.

There are three primary experimental settings in the paper: **original**, **rewrite-no-dp** and **rewrite-dp**.

An experiment using the **original** configuration can be run with `downstream_experiment_original_data.sh`.

Experiments using the **rewrite-dp** configurations can be run as follows:
- Rewriting
  - ADePT: Run with `run_adept_rewrite.sh`. Must specify a pre-trained model with `--last_checkpoint_path`, which can be prepared with the pre-training script `adept_pretrain_experiment_openwebtext_glove.sh`. When using GloVe embeddings, need to also specify a path to the embedding model with `--embed_dir_unprocessed`.
  - DP-BART-CLV: Run with `run_dp_bart_clv_rewrite.sh`. Must specify `--last_checkpoint_path`, which is the path to a pre-trained BART model.
  - DP-BART-PR: Run with `run_dp_bart_pr_rewrite.sh`. Need to include `--pruning_index_path` and `--last_checkpoint_path` from a pruned BART model. This can be prepared using `run_dp_bart_pr_pruning.sh`, which saves each checkpoint and pruned indices at every epoch.
  - DP-BART-PR+: First run additional training steps on a pruned BART model using the provided `run_dp_bart_pr_plus_training.sh` script for a given $\varepsilon$ privacy budget. Then run the rewriting script using this prepared model with `run_dp_bart_pr_plus_rewrite.sh`. Must again specify `--pruning_index_path` and `--last_checkpoint_path` from the prepared model.
- Downstream
  - The above rewriting scripts will output a rewritten dataset at the specified $\varepsilon$ privacy budget in the output directory `--output_dir`. Downstream experiments can then be run with the `run_downstream.sh` script, specifying the rewritten training dataset split with `--custom_train_path`, optionally a rewritten validation split with `--custom_valid_path` and the path to the *original* test set with `--custom_test_path`. Optionally can specify the test set with `--downstream_test_data` (e.g. `atis`, `imdb`, etc.).

Experiments using the **rewrite-no-dp** configurations can be run as above for the **rewrite-dp** configurations, but specifying `--private False` for any rewriting scripts.

Additional hyperparameters such as `--batch_size`, `--learning_rate`, and `--delta` can be modified in the script arguments. A full list of arguments and their description can be found in `settings.py`.

This repository is an extension to the framework provided at https://github.com/trusthlt/dp-rewrite. More details on running other types of configurations can be found there.

### Model Checkpoints

Checkpoints for the DP-BART-PR and DP-BART-PR+ set of models that were used in the paper are available at: https://huggingface.co/TrustHLT
