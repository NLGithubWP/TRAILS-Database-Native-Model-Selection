import torch

import argparse
import calendar
import os
import time

from common.constant import Config


def default_args(parser):

    # EA sampler's parameters,
    parser.add_argument('--population_size', type=int, default=10, help="The learning rate for REINFORCE.")
    parser.add_argument('--sample_size', type=int, default=3, help="The momentum value for EMA.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--log_name', type=str, default="baseline_train_based", help="file name to store the log")

    # define search space,
    parser.add_argument('--search_space', type=str, default="mlp_sp", help='[nasbench101, nasbench201, mlp_sp]')

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="/Users/kevin/project_python/firmest_data/",
                        help='path of data and result parent folder')

    # define search space,
    parser.add_argument('--dataset', type=str, default='frappe',
                        help='cifar10, cifar100, ImageNet16-120, frappe, criteo, uci_diabetes')

    parser.add_argument('--num_labels', type=int, default=1, help='[10, 100, 120, 2, 2, 2]')

    # those are for training

    # MLP model config
    parser.add_argument('--num_layers', default=4, type=int, help='# hidden layers')
    parser.add_argument('--nfeat', type=int, default=369,
                        help='the number of features, frappe: 5500, uci_diabetes: 369, criteo: 2100000')
    parser.add_argument('--nfield', type=int, default=43,
                        help='the number of fields, frappe: 10, uci_diabetes: 43, criteo: 39')
    parser.add_argument('--nemb', type=int, default=10,
                        help='embedding size')

    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')

    parser.add_argument('--N', type=int, default=5000, help='How many arch to train')

    default_args(parser)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_file_name", args.log_name+"_"+str(ts)+".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from logger import logger
    from controller import RegularizedEASampler
    from controller.controler import SampleController
    from search_space.init_search_space import init_search_space
    from common.structure import ModelEvaData
    from utilslibs.io_tools import write_json
    from query_api.query_mlp_api import get_mlp_training_res

    search_space_ins = init_search_space(args)
    search_space_ins.load()

    # 1. seq: init the search strategy and controller,
    strategy = RegularizedEASampler(search_space_ins,
                                    population_size=args.population_size,
                                    sample_size=args.sample_size)

    sampler = SampleController(strategy)
    arch_generator = sampler.sample_next_arch()

    explored_n = 0
    model_eva = ModelEvaData()

    pre_trained_res = get_mlp_training_res(args.dataset)[args.dataset]

    # 20 epoch for frappe and diabetes, 100 for criteo
    if args.dataset == Config.Criteo:
        epoch = 100
    else:
        epoch = 20

    base_line_log = {args.dataset: {}}
    while explored_n < args.N:
        if explored_n > 0:
            sampler.fit_sampler(model_eva.model_id, model_eva.model_score, use_prue_score=True)

        explored_n += 1
        arch_id, arch_micro = arch_generator.__next__()

        _pre_res = pre_trained_res[arch_id][str(epoch)]
        valid_auc, time_usage = _pre_res["valid_auc"], _pre_res["time_since_begin"]

        # update the shared model eval res
        model_eva.model_id = str(arch_id)
        model_eva.model_score = {"valid_auc": valid_auc}

        logger.info(f" ---- explored {explored_n} models. ")

        base_line_log[args.dataset][arch_id] = train_log

    logger.info(f" Saving result to: ", f"train_baseline_{explored_n}.json", base_line_log)
    write_json(f"train_baseline_{explored_n}.json", base_line_log)
