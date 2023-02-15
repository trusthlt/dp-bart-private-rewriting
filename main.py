import numpy as np
import torch

from experiments import Experiment, PretrainExperiment, RewriteExperiment, DownstreamExperiment
from settings import Settings, parse_arguments
from utils import prepare_specific_experiment
import pdb


def main():
    # get the settings from the command line
    ss = Settings(parse_arguments())
    ss.make_dirs()

    torch.manual_seed(ss.args.seed)
    torch.cuda.manual_seed(ss.args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(ss.args.seed)

    list_of_experiments = ['adept_l1norm_pretrain', 'adept_l2norm_pretrain',
                           'adept_l1norm_rewrite', 'adept_l2norm_rewrite']

    if ss.args.experiment is not None:
        if ss.args.experiment.lower() in list_of_experiments:
            exp_name = ss.args.experiment.lower()
            ss = prepare_specific_experiment(ss, experiment=exp_name)
        else:
            print(f"Specified experiment {ss.args.experiment} is not"
                  "available, defaulting to general experiment.")
            exp_name = 'general'
    else:
        exp_name = 'general'

    if ss.args.mode == 'pretrain':
        print(f"Running {exp_name} experiment in pretraining mode...")
        exp = PretrainExperiment(ss)
    elif ss.args.mode == 'rewrite':
        print(f"Running {exp_name} experiment in rewrite mode...")
        exp = RewriteExperiment(ss)
    elif ss.args.mode == 'downstream':
        print(f"Running {exp_name} experiment in downstream mode...")
        exp = DownstreamExperiment(ss)
    else:
        raise Exception("Experiment 'mode' must be either 'pretrain', "
                        "'rewrite', or 'downstream'.")

    # Run experiment
    exp.run_experiment()


if __name__ == '__main__':
    main()
    pass
