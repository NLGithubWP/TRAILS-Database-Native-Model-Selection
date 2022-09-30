

import json
import random
import time
import traceback

import numpy as np
import local_api
from controller import sampler_register
from eva_engine import evaluator_register
from logger import logger
import search_space
import torch
import argparse

from storage import dataset


voteList = [['nas_wot', 'synflow'],
            # ['nas_wot', 'grasp', 'synflow'],
            # ['nas_wot', 'snip', 'synflow'],
            # ['grad_norm', 'nas_wot', 'synflow'],
            # ['nas_wot', 'fisher', 'synflow'],
            # ['nas_wot', 'ntk_trace', 'synflow'],
            # ['nas_wot', 'ntk_trace_approx', 'synflow'],
            # ['grad_norm', 'nas_wot', 'grasp', 'snip', 'synflow'],
            # ['nas_wot', 'fisher', 'grasp', 'snip', 'synflow'],
            # ['grad_norm', 'nas_wot', 'fisher', 'grasp', 'synflow'],
            # ['nas_wot', 'ntk_trace', 'grasp', 'snip', 'synflow'],
            # ['grad_norm', 'nas_wot', 'fisher', 'snip', 'synflow']
            ]

# singleList = ["grad_norm", "nas_wot", "grasp", "synflow", "snip", "ntk_trace", "fisher", "ntk_trace_approx"]
# singleList = ["nas_wot", "grasp", "synflow", "snip", "ntk_trace", "fisher", "ntk_trace_approx"]
singleList = ["nas_wot", "synflow"]


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # define architecture sampler, [test, random, rl, ea, bohb]
    parser.add_argument('--arch_sampler', type=str, default="ea",
                        help='which sampler to use, [test, random, rl, ea, bohb ]')
    parser.add_argument('--is_vote', type=int, default=0)
    parser.add_argument('--out_folder', type=str,
                        default="./main/3_benchmark_sampler/101_cifar10_sampling_res/", help="data set name")
    # parser.add_argument('--pre_scored_data', type=str,
    #                     default="./result_bs32_ic16/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json",
    #                     help="1")
    parser.add_argument('--pre_scored_data', type=str,
                        default="./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json", help="1")

    parser.add_argument('--gt_file', type=str, default="./101_result", help="1")

    # job config
    parser.add_argument('--log_name', type=str, default="result.json", help="log_name")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="./data",
                        help='path of data folder')

    # dataLoader setting,
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='in one of [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--num_data_workers', type=int, default=1,
                        help='number of workers for dataLoader')
    parser.add_argument('--batch_sample_alg', type=str, default="random",
                        help='sample a mini batch, [random, grasp]')

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench101",
                        help='which search space to use, [nasbench101, nasbench201, ... ]')
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

    parser.add_argument('--total_run', type=int, default=300,
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


def evaluate_arch(evaluator, arch_id, used_search_space, args, mini_batch, mini_batch_targets):
    new_bn = True
    if evaluator.__class__.__name__ == "SynFlowEvaluator":
        new_bn = False

    arch_id = int(arch_id)
    final_score = 0
    final_cost = 0

    new_arch = used_search_space.copy_architecture(arch_id, None, new_bn)
    try:
        new_arch = new_arch.to(args.device)
        score, time_usage = evaluator.evaluate_wrapper(
            arch=new_arch,
            device=args.device,
            batch_data=mini_batch,
            batch_labels=mini_batch_targets)
        logger.info(evaluator.__class__.__name__ + ":" + str(score))
        final_score += score
        final_cost += time_usage
    except Exception as e:
        if "out of memory" in str(e):
            logger.info("======================================================")
            logger.error("architecture " + str(arch_id) + " will be evaluate in CPU, message = " + str(e))
            logger.info("======================================================")
            mini_batch_cpu = mini_batch.cpu()
            mini_batch_targets_cpu = mini_batch_targets.cpu()
            new_arch_cpu = used_search_space.copy_architecture(arch_id, new_arch.cpu(), new_bn).cpu()
            score, cpu_time_usage = evaluator.evaluate_wrapper(
                arch=new_arch_cpu,
                device="cpu",
                batch_data=mini_batch_cpu,
                batch_labels=mini_batch_targets_cpu)
            logger.info(evaluator.__class__.__name__ + " on cpu:" + str(score))
            final_score += score
            final_cost += cpu_time_usage
            del new_arch_cpu
        else:
            logger.info("======================================================")
            logger.error(traceback.format_exc())
            logger.error("error when evaluate architecture " + str(arch_id) + ", message = " + str(e))
            logger.info("======================================================")
            exit(0)
    finally:
        # clean old graph
        del new_arch
        torch.cuda.empty_cache()

    # 3. Query to query the real performance
    return final_score, final_cost


def get_score_result(arch_id, loapi, alg_name, used_search_space, args, mini_batch, mini_batch_targets):
    if loapi.is_arch_inside_data(arch_id, alg_name):
        score = loapi.api_get_score(arch_id, alg_name)
    else:
        score, _ = evaluate_arch(evaluator_register[alg_name], int(arch_id), used_search_space, args, mini_batch, mini_batch_targets)
        loapi.update_existing_data(arch_id, alg_name, score)
    return score


if __name__ == '__main__':

    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    args = parse_arguments()

    if args.search_space == "nasbench101":
        args.api_loc = "nasbench_only108.pkl"
    elif args.search_space == "nasbench201":
        args.api_loc = "NAS-Bench-201-v1_0-e61699.pth"

    if args.is_vote == 1:
        d_is_vote_str = "vote"
    else:
        d_is_vote_str = "single"

    args.save_all_run_file = args.out_folder+"{}_{}_{}".format(args.arch_sampler, d_is_vote_str, args.dataset)
    logger.info("running with params: :" + json.dumps(args.__dict__, indent=2))

    # define dataLoader, and sample a mini-batch
    train_loader, val_loader, class_num = dataset.get_dataloader(
        train_batch_size=1,
        test_batch_size=1,
        dataset=args.dataset,
        num_workers=args.num_data_workers,
        datadir=args.base_dir)
    args.num_labels = class_num
    if args.batch_size // class_num == 0:
        logger.info("batch_size is smaller than class_num", args.batch_size, class_num )
        # exit(0)
    # sample a batch with random or GRASP
    mini_batch, mini_batch_targets = dataset.get_mini_batch(
        dataloader=train_loader, sample_alg=args.batch_sample_alg, batch_size=args.batch_size, num_classes=class_num)
    mini_batch = mini_batch.to(args.device)
    mini_batch_targets = mini_batch_targets.to(args.device)

    used_search_space = search_space.init_search_space(args)

    loapi = local_api.LocalApi(args.pre_scored_data, args.gt_file, used_search_space, args.search_space)

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
                    arch_generate_time = time.time()
                    arch_id, _ = arch_generator.__next__()
                    print("architecture generating time = ", time.time() - arch_generate_time)
                    _num_arch_each_run += 1

                    rank_score = 0.0
                    gt = loapi.api_get_ground_truth(arch_id, args.dataset)
                    # get final score with multiple votes.
                    for alg_name in vote_com:
                        score_ = get_score_result(arch_id, loapi, alg_name, used_search_space, args, mini_batch, mini_batch_targets)
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

                sampler_begin = time.time()
                sampler = sampler_register[args.arch_sampler](used_search_space, args)
                arch_generator = sampler.sample_next_arch(args.arch_size)
                # print("time for generate sampler = ", time.time() - sampler_begin)

                _num_arch_each_run = 0
                while _num_arch_each_run < args.num_arch_each_run:

                    gen_arch_begin = time.time()
                    arch_id, _ = arch_generator.__next__()
                    # print("time for generate new architecture = ", _num_arch_each_run, time.time() - gen_arch_begin)
                    _num_arch_each_run += 1

                    score_arch_begin = time.time()
                    gt = loapi.api_get_ground_truth(arch_id, args.dataset)
                    score = get_score_result(arch_id, loapi, alg_name, used_search_space, args, mini_batch, mini_batch_targets)

                    # record acc, score
                    alg_map[alg_name]["arch_id"].append(arch_id)
                    alg_map[alg_name]["ori_score"].append(score)
                    alg_map[alg_name]["acc"].append(gt)

                    # use rank to fit the sampler.
                    fit_time = time.time()
                    rank = loapi.get_rank_score(alg_map[alg_name]["ori_score"])
                    sampler.fit_sampler(rank)
                    # print("time for fit sampler = ", _num_arch_each_run, time.time() - fit_time)

                all_run_result[run_id][alg_name] = alg_map[alg_name]

        print("time takes = {}".format(time.time() - run_begin_time))

    loapi.save_latest_data()

    with open(args.save_all_run_file, 'w') as outfile:
        outfile.write(json.dumps(all_run_result))
