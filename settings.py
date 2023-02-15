import os
import pathlib
import time
import argparse
import re
from utils import get_model_type
import pdb

project_root = os.path.join(pathlib.Path(__file__).parent.resolve())


def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected boolean value.')


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def parse_arguments():
    argparser = argparse.ArgumentParser()

    # Main parameters
    argparser.add_argument("--mode", type=str, default='pretrain',
                           help='Mode of experiment (pretrain, rewrite, downstream).')
    argparser.add_argument("--dataset", type=str, required=True,
                           help='mode=pretrain: Which dataset will be used for training the autoencoder. '
                                'mode=rewrite: Which dataset will be rewritten. '
                                'mode=downstream: Which dataset the downstream task will be trained on.')
    argparser.add_argument("--model", type=str, default='adept',
                           help="Model to run, must match the specified mode (see list of models in README). (Currently only have 'adept' for pre-training/rewriting and 'bert_downstream' for downstream classification.)")

    # optional, general params: meta params for running experiments
    argparser.add_argument("--name", type=str, default=None,
                           help='The experiment name. Defaults to the current timestamp.')
    argparser.add_argument("--experiment", type=str, default=None,
                           help='The type of experiment to be run, can either choose from an existing set available in the project, a custom name for a custom experiment, or left as None, in which case the default experiment pipeline will be run.')
    argparser.add_argument("--seed", type=int, default=12345)
    argparser.add_argument("--test_only", type=str2bool, nargs='?',
                           const=True, default=False,
                           help='Only test on saved downstream model '
                                'according to specified configuration '
                                '(only applies to mode=downstream).'
                                'If you set this to true, the experiment name and results directory need to have a checkpoint available to load from.')
    argparser.add_argument('--local', type=str2bool, nargs='?', const=True,
                           default=False, help='Run the experiment in a'
                                               'limited setting for local'
                                               'debugging (with CPU and limited'
                                               'iteration size, defined with'
                                               '"local_iter_size" argument).')
    argparser.add_argument('--local_iter_size', type=int, default=10,
                           help='Determines number of iterations per epoch'
                           'when running experiments locally.')

    # optional, general params: directories
    argparser.add_argument("--output_dir", type=str, default=os.path.join(project_root, 'results'),
                           help='Where stats & logs will be saved (a subfolder will be created for '
                                'the experiment). Defaults to <project_root>/results')
    argparser.add_argument("--dump_dir", type=str, default=None,
                           help='Where stuff that might need much storage (e.g., model checkpoints & rewritten data) '
                                'will be saved. Defaults to output_dir.')

    argparser.add_argument("--asset_dir", type=str, default=os.path.join(project_root, 'assets'),
                           help='Where to look for assets like data sets & embeddings. '
                                'Defaults to <project_root>/assets. '
                                'For data sets, this can be overwritten with the data_dir argument.')
    argparser.add_argument("--data_dir", type=str, default=None,
                           help='Where to look for data sets. Defaults to asset_dir')
    argparser.add_argument("--embed_dir_unprocessed", type=str, default=None,
                           help='Where to look for pre-trained embedding models (e.g. Word2Vec, GloVe).')

    argparser.add_argument("--custom_train_path", type=str, default=None, help='Where to look for a custom datasets (train partition), if not using one of the prepared datasets for the framework.')
    argparser.add_argument("--custom_valid_path", type=str, default=None, help='Where to look for a custom datasets (optional validation partition), if not using one of the prepared datasets for the framework. If not supplied for pre-training, validation set will be created from a random split of the train partition.')
    argparser.add_argument("--custom_test_path", type=str, default=None, help='Where to look for a custom datasets (optional test partition), if not using one of the prepared datasets for the framework.')
    argparser.add_argument("--downstream_test_data", type=str, default=None, help='Will determine a test set to load from one of the datasets of the framework (downstream mode only, when loading a training/validation dataset from custom paths).')

    # optional, general params: datasets
    argparser.add_argument("--prepend_labels", type=str2bool, nargs='?',
                           const=True, default=False,
                           help='Whether to prepend labels to the beginning'\
                                'of each sequence.')
    argparser.add_argument("--train_ratio", type=float, default=0.8,
                           help='Training dataset size ratio for train/validation split (pretrain/downstream modes only). Not used if custom dataset specified with a path for the validation set.')
    argparser.add_argument("--length_threshold", type=int, default=None,
                           help='An optional value by which to subsample a dataset to only include document that are originally below a certain number of tokens (based on a split by whitespace for each document).')
    argparser.add_argument("--custom_preprocessor", type=str2bool, nargs='?',
                           const=True, default=False, help='Whether to use a custom preprocessor instead of one of the built-in ones.')
    argparser.add_argument("--last_checkpoint_path", type=str, default='',
                           help='Global path of the checkpoint that should be used. '
                                'Pretrain mode will use this to resume training. '
                                'Rewrite mode will load the model used for rewriting from that checkpoint.')
    argparser.add_argument("--save_unique_checkpoint_every_epoch", type=str2bool, nargs='?',
                           const=True, default=False, help='If True, will save a unique checkpoint for every epoch of pretraining (pretrain mode) or downstream training (downstream mode). If False, will only save the latest checkpoint.')
    argparser.add_argument("--include_original", type=str2bool, nargs='?', const=True, default=False, help='Whether or not to include the original dataset in the rewritten dataframe (rewrite mode only).')
    argparser.add_argument("--data_split_cutoff", type=int, default=None, help='If not "None", will use a subset of the "train" data split based on the specified int value (e.g. 10000 to only use the first 10000 data points). Note: This is applied before any train-validation splits, in case validation data is not present.')
    argparser.add_argument("--iteration_cutoff", type=int, default=None, help='If not "None", will only run training and validation for the specified amount of iterations every epoch (pretraining and downstream modes). Note: "--local_iter_size" supercedes this value when running experiments locally.')

    # optional, general params: hyperparams
    argparser.add_argument("--epochs", type=int, default=5)
    argparser.add_argument("--batch_size", type=int, default=1)
    argparser.add_argument("--learning_rate", type=float, default=0.001)
    argparser.add_argument("--weight_decay", type=float, default=0.00)
    argparser.add_argument("--early_stopping", type=str2bool, nargs='?',
                           const=True, default=True)
    argparser.add_argument("--patience", type=int, default=20)
    argparser.add_argument("--optim_type", type=str, default='adam',
                           help='sgd or adam')
    argparser.add_argument("--two_optimizers", type=str2bool, nargs='?',
                           const=True, default=True, help='Requires for the specified model to have distinct "encoder" and "decoder" components.')
    argparser.add_argument("--hidden_size", type=int, default=768)
    argparser.add_argument("--enc_out_size", type=int, default=128,
                           help='Specific to RNN models.')
    argparser.add_argument("--vocab_size", type=int, default=20000,
                           help='Specific to RNN models, specifies the maximum vocabulary size based on frequency when using a pre-trained word embedding model.')
    argparser.add_argument("--max_seq_len", type=int, default=512)
    argparser.add_argument("--embed_size", type=int, default=300)
    argparser.add_argument("--embed_type", type=str, default='glove',
                           help='"glove" or "word2vec"')
    argparser.add_argument("--transformer_type", type=str, default='bert-base-uncased')
    argparser.add_argument("--pruning", type=str2bool, nargs='?', const=True, default=False, help='Whether to prune neurons or not during the pre-training procedure, iteratively increasing the amount of pruning per k iterations (specified below `--prune_finetune_k`), or prune using specified pruning indexes during rewriting (specified below with `--pruning_index_path`).')
    argparser.add_argument("--gradually_increase_pruning", type=str2bool, nargs='?', const=True, default=True, help='Whether to increase the amount of pruned neurons after every epoch, or maintain a constant number throughout.')
    argparser.add_argument("--prune_finetune_k", type=int, default=1000, help='Number of iterations to fine-tune, before pruning again when `--pruning` is True.')
    argparser.add_argument("--pruning_index_path", type=str, default='', help='Path for loading the pruning indexes during rewriting.')
    argparser.add_argument("--discretize", type=str2bool, nargs='?', const=True, default=False, help='Whether to discretize encoder output vectors for the DP-BART model. Note this procedure is included when selecting the `GRR` DP module.')
    argparser.add_argument("--train_teacher_forcing_ratio", type=float,
                           default=0.0, help='For RNN-based models.')
    argparser.add_argument("--private", type=str2bool, nargs='?',
                           const=True, default=False,
                           help='If privatization should be applied during pre-training')
    argparser.add_argument("--epsilon", type=float, default=1)
    argparser.add_argument("--delta", type=float, default=1e-5)
    argparser.add_argument("--clipping_constant", type=float, default=1.)
    argparser.add_argument("--save_initial_model", type=str2bool, nargs='?', const=True, default=False, help='Whether to save a checkpoint of the model before starting the training procedure (pre-training mode only). For convenient comparison of rewriting capabilities for models that were not locally pre-trained. Model will not be saved if loading from an existing checkpoint.')
    argparser.add_argument("--dp_mechanism", type=str, default='laplace', help='laplace or gaussian.'
                            'Has no effect as of now.')
    argparser.add_argument("--dp_module", type=str, default='clip_norm', help='The type of DP module to be applied, specific to certain autoencoder models. Relevant arguments to be specified for each specific module.')
    argparser.add_argument("--l_norm", type=int, default=2, help='Pass 2 for L2 norm, 1 for L1 norm.'
                            'Original adept uses the L2-norm (default value of this param) '
                            '(which is one of the reasons why Adept is not DP, cf. '
                            'http://dx.doi.org/10.18653/v1/2021.emnlp-main.114). Used in the "clip_norm" dp_module')
    argparser.add_argument("--no_clipping", type=str2bool, nargs='?', const=True, default=False, help='Whether or not to clip encoder hidden states in the non-private setting.')
    argparser.add_argument("--custom_model_arguments", nargs='*', help='Additional optional arguments for a custom model, no upper limit on the number.')

    args = argparser.parse_args()

    return args


class Settings(object):
    '''
    Configuration for the project.
    '''
    def __init__(self, args):
        # args, e.g. the output of settings.parse_arguments()
        self.args = args

        now = time.localtime()
        self.args.formatted_now = f'{now[0]}-{now[1]}-{now[2]}--{now[3]:02d}-{now[4]:02d}-{now[5]:02d}'

        ## Determining model type
        self.args.model_type = get_model_type(self.args.model)

        if self.args.name is None or self.args.name == '':
            self.args.name = self.args.formatted_now
        if self.args.dump_dir is None:
            self.args.dump_dir = self.args.output_dir
        if self.args.data_dir is None:
            self.args.data_dir = self.args.asset_dir

        self.exp_output_dir = os.path.join(self.args.output_dir, self.args.name)
        self.exp_dump_dir = os.path.join(self.args.dump_dir, self.args.name)
        self.checkpoint_dir = os.path.join(self.exp_dump_dir, 'checkpoints')
        self.embed_dir_processed = None

    def make_dirs(self):
        for d in [
            self.args.output_dir,
            self.args.dump_dir,
            self.args.asset_dir,
            self.args.data_dir,
            self.exp_output_dir,
            self.exp_dump_dir,
            self.checkpoint_dir
            ]:
            if not os.path.exists(d):
                os.makedirs(d)

        if self.args.model_type == 'rnn':
            self.embed_dir_processed = os.path.join(self.args.asset_dir,
                                                    'embeds')
            if not os.path.exists(self.embed_dir_processed):
                os.makedirs(self.embed_dir_processed)
