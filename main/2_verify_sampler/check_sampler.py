

import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import local_api
from sampler import sampler_register
import search_space
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    parser.add_argument('--dataset', type=str, default="cifar10", help="log_name")
    # job config
    parser.add_argument('--log_name', type=str, default="result.json", help="log_name")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="./data",
                        help='path of data folder')

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench101",
                        help='which search space to use, [nasbench101, nasbench201, ... ]')

    parser.add_argument('--api_loc', type=str, default="nasbench_only108.pkl",
                        help='which search space to use, ['
                             'nasbench101: nasbench_only108.pkl '
                             'nasbench201: NAS-Bench-201-v1_0-e61699.pth'
                             ' ... ]')
    # define device
    parser.add_argument('--device', type=str, default="cpu",
                        help='which device to use [cpu, cuda]')

    # search space configs for nasBench101
    parser.add_argument('--init_channels', default=16, type=int, help='output channels of stem convolution')
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')

    #  weight initialization settings, nasBench201 requires it.
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')

    parser.add_argument('--bn', type=int, default=1,
                        help="If use batch norm in network 1 = true, 0 = false")

    parser.add_argument('--arch_size', type=int, default=7,
                        help='How many node the architecture has at least')

    parser.add_argument('--img_name', type=str, default="check_sampler.jpg",
                        help='')

    parser.add_argument('--gt_file', type=str, default="./101_result", help="1")

    parser.add_argument('--num_labels', type=int, default=10, help="class number ")

    # RL sampler's parameters,
    parser.add_argument('--rl_learning_rate', type=float, help="The learning rate for REINFORCE.", default=0.0001)
    parser.add_argument('--rl_EMA_momentum', type=float, default=0.9, help="The momentum value for EMA.")

    # EA sampler's parameters,
    parser.add_argument('--population_size', type=int, default=10, help="The learning rate for REINFORCE.")
    parser.add_argument('--sample_size', type=int, default=3, help="The momentum value for EMA.")

    parser.add_argument('--total_run', type=int, default=500,
                        help='Total run number in benchmark stage. ')

    parser.add_argument('--num_arch_each_run', type=int, default=1000,
                        help='How many arch to evaluate in each run')

    parser.add_argument('--save_file', type=str, default="compare_sampler_file",
                        help='How many arch to evaluate in each run')

    parser.add_argument('--save_img', type=str, default="compare_sampler_file.jpg",
                        help='How many arch to evaluate in each run')

    return parser.parse_args()


def plot_experiment(exp, label):
    exp = np.array(exp)
    q_75 = np.quantile(exp, .75, axis=0)
    q_25 = np.quantile(exp, .25, axis=0)
    mean = np.quantile(exp, .5, axis=0)
    plt.plot(mean, label=label)
    plt.fill_between(range(len(q_25)), q_25, q_75, alpha=0.3)
    plt.grid()


def draw_sampler_res_sub(all_RUN_result, image_name):

    index = 0
    for sampler_name in list(all_RUN_result[list(all_RUN_result.keys())[0]].keys()):
        score_list = []
        for run_id_m in all_RUN_result:
            score_list.append(all_RUN_result[run_id_m][sampler_name])
        plot_experiment(score_list, sampler_name)
        index += 1

    plt.legend()
    plt.ylim(0.93, 0.945)
    plt.tight_layout()
    plt.title(image_name)
    # plt.show()
    plt.savefig(image_name, bbox_inches='tight')


if __name__ == '__main__':

    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    args = parse_arguments()

    used_search_space = search_space.init_search_space(args)

    loapi = local_api.LocalApi(None, args.gt_file, used_search_space, args.search_space)

    sampler_name_list = []
    for sampler_name, _ in sampler_register.items():
        sampler_name_list.append(sampler_name)
    # sampler_name_list = [CommonVars.EA_SAMPLER]

    all_RUN_result = {}
    # for each run
    for run_id in range(args.total_run):

        all_RUN_result[run_id] = {}
        begin_run_time = time.time()
        # for each metrics, fit different sampler.
        for sampler_name in sampler_name_list:
            print("run id", run_id, "sampler_name", sampler_name)
            # init sampler
            sampler = sampler_register[sampler_name](used_search_space, args)
            arch_generator = sampler.sample_next_arch(args.arch_size)

            plot_test_accuracy = []
            for _ in range(1, args.num_arch_each_run):
                arch_id, _ = arch_generator.__next__()
                acc = loapi.api_get_ground_truth(arch_id, args.dataset)

                # record the best architecture up to now
                if len(plot_test_accuracy) > 0:
                    if acc > plot_test_accuracy[-1]:
                        plot_test_accuracy.append(acc)
                    else:
                        plot_test_accuracy.append(plot_test_accuracy[-1])
                else:
                    plot_test_accuracy.append(acc)

                # fit the sampler
                sampler.fit_sampler(acc)

            if sampler_name not in all_RUN_result[run_id]:
                all_RUN_result[run_id][sampler_name] = []

            all_RUN_result[run_id][sampler_name] = plot_test_accuracy

        print("Total time usage = ", time.time() - begin_run_time)

    with open('./'+args.save_file, 'w') as outfile:
        outfile.write(json.dumps(all_RUN_result))
    draw_sampler_res_sub(all_RUN_result, args.save_img)

