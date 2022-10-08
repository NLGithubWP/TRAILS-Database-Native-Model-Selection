

import json
import random
import time
import traceback
import numpy as np
import os
import torch
import argparse
import calendar
from query_api.gt_api import Gt201, Gt101
from common.constant import Config
from controller.controler import Controller
from utilslibs.tools import write_json

base_dir = os.getcwd()


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument(
        '--metrics_result', type=str,
        default=os.path.join(base_dir, "result_base/result_system/simulate/train_based_201_imgNet_100run_3km_ea.json"),
        help="metrics_result")

    # job config
    parser.add_argument('--num_run', type=int, default=100, help="num of run")
    parser.add_argument('--num_arch', type=int, default=15625, help="how many architecture to evaluate")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default=os.path.join(base_dir, "data"), help='path of data folder')

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench201",
                        help='search space to use, [nasbench101, nasbench201, ... ]')

    parser.add_argument('--api_loc', type=str, default="NAS-Bench-201-v1_1-096897.pth",
                        help='which search space to use, ['
                             'nasbench101: nasbench_only108.pkl'
                             'nasbench201: NAS-Bench-201-v1_0-e61699.pth'
                             'nasbench201: NAS-Bench-201-v1_1-096897.pth'
                             ' ... ]')

    parser.add_argument('--dataset', type=str, default='ImageNet16-120', help='[cifar10, cifar100, ImageNet16-120]')

    # define device
    parser.add_argument('--init_channels', default=16, type=int, help='output channels of stem convolution')
    # search space configs for nasBench101
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')

    #  weight initialization settings, nasBench201 requires it.
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')

    parser.add_argument('--bn', type=int, default=1,
                        help="If use batch norm in network 1 = true, 0 = false")

    # nas101 doesn't need this, while 201 need it.
    parser.add_argument('--arch_size', type=int, default=1,
                        help='How many node the architecture has at least')

    # EA sampler's parameters,
    parser.add_argument('--population_size', type=int, default=10, help="The learning rate for REINFORCE.")
    parser.add_argument('--sample_size', type=int, default=3, help="The momentum value for EMA.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    args.num_labels = 10
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_file_name", args.metrics_result[:-5]+"_"+str(ts)+".log" )

    from logger import logger
    import search_space

    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    logger.info("running with params: :" + json.dumps(args.__dict__, indent=2))
    logger.info("cuda available = " + str(torch.cuda.is_available()))

    used_search_space = search_space.init_search_space(args)

    all_run_info = {}
    for run_id in range(args.num_run):
        run_begin_time = time.time()
        # 1. Sampler one architecture
        sampler = Controller(used_search_space, args)
        arch_generator = sampler.sample_next_arch(args.arch_size)

        # for logging result
        arch_id_list = []
        current_best_acc = []
        tfree_best = 0
        x_axis_time = []
        current_x_time_usage = 0

        total_fit_time = 0
        total_compute_time = 0

        i = 1
        try:
            while True:
                if i > args.num_arch:
                    break
                begin_run_time = time.time()
                # new arch
                arch_id, _ = arch_generator.__next__()
                arch_id_list.append(arch_id)

                begin_get_score = time.time()
                # simulate training
                test_accuracy = 0
                time_usage = 0
                if args.search_space == Config.NB201:
                    data201 = Gt201()
                    test_accuracy, time_usage = data201.query_200_epoch(arch_id=str(arch_id), dataset=args.dataset)
                elif args.search_space == Config.NB101:
                    data101 = Gt101()
                    test_accuracy, time_usage = data101.get_c10_test_info(arch_id=arch_id)
                else:
                    exit(1)
                total_compute_time += time.time() - begin_get_score

                # fit sampler
                begin_fit = time.time()
                sampler.search_strategy.fit_sampler(test_accuracy)
                # print("fit time", time.time() - begin_fit)

                # record time and acc
                if test_accuracy > tfree_best:
                    tfree_best = test_accuracy
                current_x_time_usage += time_usage

                current_best_acc.append(tfree_best)
                x_axis_time.append(current_x_time_usage)

                i = i + 1
                # print(time.time() - begin_run_time)
            all_run_info[run_id] = {}
            all_run_info[run_id]["arch_id_list"] = arch_id_list
            all_run_info[run_id]["current_best_acc"] = current_best_acc
            all_run_info[run_id]["x_axis_time"] = x_axis_time

        except Exception as e:
            logger.info("========================================================================")
            logger.error(traceback.format_exc())
            logger.error("error: " + str(e))
            logger.info("========================================================================")
            exit(1)

        print("run {} finished using {}".format(run_id, time.time() - run_begin_time))
        logger.info("run {} finished using {}".format(run_id, time.time() - run_begin_time))

    write_json(args.metrics_result, all_run_info)

