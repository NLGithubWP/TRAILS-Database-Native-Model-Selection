

import json
import random
import time

import numpy as np
from controller import sampler_register
from logger import logger
import search_space
import torch
import argparse

from query_api.query_p1_score_api import LocalApi

voteList = [['nas_wot', 'synflow'],
            ]

singleList = ["grad_norm", "nas_wot", "grasp", "synflow", "snip", "ntk_trace", "fisher", "ntk_trace_approx"]


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # define architecture sampler, [test, random, rl, ea, bohb]
    parser.add_argument('--arch_sampler', type=str, default="rl",
                        help='which sampler to use, [test, random, rl, ea, bohb ]')
    parser.add_argument('--is_vote', type=int, default=0)
    parser.add_argument('--out_folder', type=str, default="./", help="data set name")
    parser.add_argument('--pre_scored_data', type=str,
                        default="./result_bs32_ic16/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json",
                        help="log_name")
    parser.add_argument('--gt_file', type=str, default="./101_result", help="log_name")
    parser.add_argument('--num_labels', type=int, default=10, help="class number ")
    parser.add_argument('--dataset', type=str, default="cifar10", help="data set name")

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

    parser.add_argument('--total_run', type=int, default=100,
                        help='Total run number in benchmark stage. ')

    parser.add_argument('--num_arch_each_run', type=int, default=400,
                        help='How many arch to evaluate in each run')

    # RL sampler's parameters,
    parser.add_argument('--rl_learning_rate', type=float, help="The learning rate for REINFORCE.", default=0.0001)
    parser.add_argument('--rl_EMA_momentum', type=float, default=0.9, help="The momentum value for EMA.")
    # EA sampler's parameters,
    parser.add_argument('--population_size', type=int, default=10, help="The learning rate for REINFORCE.")
    parser.add_argument('--sample_size', type=int, default=3, help="The momentum value for EMA.")

    return parser.parse_args()


if __name__ == '__main__':

    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    args = parse_arguments()
    loapi = LocalApi()

    if args.is_vote == 1:
        d_is_vote_str = "vote"
    else:
        d_is_vote_str = "single"

    args.save_all_run_file = args.out_folder+"{}_{}_{}".format(args.arch_sampler, d_is_vote_str, args.dataset)
    logger.info("running with params: :" + json.dumps(args.__dict__, indent=2))
    used_search_space = search_space.init_search_space(args)

    all_run_result = {}
    for run_id in range(args.total_run):
        print("run id", run_id)
        run_begin_time = time.time()

        all_run_result[run_id] = {}
        # if this is vote
        if args.is_vote == 1:
            alg_map = {}
            for vote_com in voteList:
                vote_comb_name = "_".join(vote_com)
                # init dict
                alg_map[vote_comb_name] = {}
                alg_map[vote_comb_name]["ori_score"] = []
                alg_map[vote_comb_name]["acc"] = []
                alg_map[vote_comb_name]["arch_id"] = []
                for alg_name in vote_com:
                    alg_map[alg_name] = {}
                    alg_map[alg_name]["ori_score"] = []
                # init sampler
                sampler = sampler_register[args.arch_sampler](used_search_space, args)
                arch_generator = sampler.sample_next_arch(args.arch_size)

                _num_arch_each_run = 0
                while _num_arch_each_run < args.num_arch_each_run:
                    arch_id, _ = arch_generator.__next__()
                    # if arch_id not in loapi.data:
                    #     continue
                    _num_arch_each_run += 1

                    gt = loapi.api_get_ground_truth(arch_id, args.dataset)

                    rank_score = 0.0
                    # get final score with multiple votes.
                    for alg_name in vote_com:
                        score_ = loapi.api_get_score(arch_id, alg_name)
                        alg_map[alg_name]["ori_score"].append(score_)
                        rank_ = loapi.get_rank_score(alg_map[alg_name]["ori_score"])
                        rank_score += rank_

                    # record acc, score
                    alg_map[vote_comb_name]["acc"].append(gt)
                    alg_map[vote_comb_name]["arch_id"].append(arch_id)
                    alg_map[vote_comb_name]["ori_score"].append(rank_score)

                    rank = loapi.get_rank_score(alg_map[vote_comb_name]["ori_score"])
                    sampler.fit_sampler(rank)

                all_run_result[run_id][vote_comb_name] = alg_map[vote_comb_name]
        # this is not vote
        else:
            alg_map = {}
            for alg_name in singleList:

                # tmp map
                alg_map[alg_name] = {}
                alg_map[alg_name]["ori_score"] = []
                alg_map[alg_name]["acc"] = []
                alg_map[alg_name]["arch_id"] = []

                sampler = sampler_register[args.arch_sampler](used_search_space, args)
                arch_generator = sampler.sample_next_arch(args.arch_size)

                _num_arch_each_run = 0
                while _num_arch_each_run < args.num_arch_each_run:
                    arch_id, _ = arch_generator.__next__()
                    # if arch_id not in loapi.data:
                    #     continue
                    _num_arch_each_run += 1
                    score = loapi.api_get_score(arch_id, alg_name)
                    gt = loapi.api_get_ground_truth(arch_id, args.dataset)

                    # record acc, score
                    alg_map[alg_name]["arch_id"].append(arch_id)
                    alg_map[alg_name]["ori_score"].append(score)
                    alg_map[alg_name]["acc"].append(gt)

                    # use rank to fit the sampler.
                    rank = loapi.get_rank_score(alg_map[alg_name]["ori_score"])
                    sampler.fit_sampler(rank)

                all_run_result[run_id][alg_name] = alg_map[alg_name]

        print("time takes = {}".format(time.time() - run_begin_time))

    with open(args.save_all_run_file, 'w') as outfile:
        outfile.write(json.dumps(all_run_result))
