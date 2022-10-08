import argparse
import os
import random

import numpy as np
from matplotlib import pyplot as plt

import search_space

from common.constant import Config
from eva_engine.run_ms import RunModelSelection
from query_api.parse_pre_res import FetchGroundTruth


def default_args(parser):
    parser.add_argument('--init_channels', default=16, type=int, help='output channels of stem convolution')
    # search space configs for nasBench101
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='# modules per stack')

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

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default=os.path.join(base_dir, "data"),
                        help='path of data folder')


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--controller_port', type=int, default=8002)
    parser.add_argument('--log_name', type=str, default="SS_1wk_1run_NB101_c10.log")

    parser.add_argument('--run', type=int, default=100, help="how many run")
    parser.add_argument('--save_file_latency', type=str,
                        default=os.path.join(base_dir, "1wk_1run_NB101_c10_latency"), help="search target")
    parser.add_argument('--save_file_all', type=str,
                        default=os.path.join(base_dir, "1wk_1run_NB101_c10_all"), help="search target")

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench201",
                        help='search space to use, [nasbench101, nasbench201, ... ]')

    # define search space,
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--num_labels', type=int, default=10, help='[10, 100, 120]')

    parser.add_argument('--api_loc', type=str, default="NAS-Bench-201-v1_1-096897.pth",
                        help='which search space to use, ['
                             'nasbench101: nasbench_only108.pkl'
                             'nasbench201: NAS-Bench-201-v1_1-096897.pth'
                             ' ... ]')

    default_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    base_dir = os.getcwd()
    random.seed(10)
    args = parse_arguments()

    budget_array = [1, 3, 5, 7, 10, 30, 50, 70, 100, 300, 500, 700, 1000, 1300, 1440]
    dataset = Config.NB201
    fgt = FetchGroundTruth(args.search_space)
    used_search_space = search_space.init_search_space(args)

    y_acc_list_arr = []
    x_T_list = budget_array
    real_time_used_arr = []
    planed_time_used_arr = []

    for run_id in range(100):
        y_each_run = []
        real_time_ech_run = []
        plan_time_ech_run = []
        for T in budget_array:
            best_arch, acc_sh_v, real_time_used, plan_used = \
                RunModelSelection(args, fgt, used_search_space).select_model(T*60, run_id)
            y_each_run.append(acc_sh_v)
            real_time_ech_run.append(real_time_used)
            plan_time_ech_run.append(plan_used)
        y_acc_list_arr.append(y_each_run)
        real_time_used_arr.append(real_time_ech_run)
        planed_time_used_arr.append(plan_time_ech_run)

    print(y_acc_list_arr)
    print(x_T_list)
    print(real_time_used_arr)
    print(planed_time_used_arr)

    exp = np.array(y_acc_list_arr)
    q_75 = np.quantile(exp, .75, axis=0)
    q_25 = np.quantile(exp, .25, axis=0)
    mean = np.quantile(exp, .5, axis=0)
    # plot simulate result
    plt.fill_between(budget_array, q_25, q_75, alpha=0.1)
    plt.plot(x_T_list, mean, "-*", label="FastAutoNAS")

    plt.xscale("symlog")
    plt.grid()
    plt.xlabel("Time Budget given by user (mins)")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.show()



