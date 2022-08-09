

import json
import random
import time
import traceback
import numpy as np
from common.constant import CommonVars
from logger import logger
from search_algorithm.utils.gpu_util import showUtilization
from storage import dataset
from sampler import sampler_register
from search_algorithm import evaluator_register
import search_space
import torch
import argparse
import scipy.stats as ss


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--log_name', type=str, default="result.json", help="log_name")

    # job config
    parser.add_argument('--num_arch', type=int, default=2, help="how many architecture to evaluate")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="/Users/kevin/project_python/Fast-AutoNAS/data",
                        help='path of data folder')

    # dataLoader setting,
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='in one of [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--num_data_workers', type=int, default=1,
                        help='number of workers for dataLoader')
    parser.add_argument('--batch_sample_alg', type=str, default="random",
                        help='sample a mini batch, [random, grasp]')

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench201",
                        help='which search space to use, [nasbench101, nasbench201, ... ]')

    parser.add_argument('--api_loc', type=str, default="NAS-Bench-201-v1_0-e61699.pth",
                        help='which search space to use, ['
                             'nasbench101: nasbench_only108.pkl '
                             'nasbench201: NAS-Bench-201-v1_0-e61699.pth'
                             ' ... ]')

    # define architecture sampler,
    parser.add_argument('--arch_sampler', type=str, default="random",
                        help='which architecture sampler to use, [test, random, BOHB, ...]')

    parser.add_argument('--time_budget', type=int, default=500,
                        help='How many time (second) to use in sampling & scoring')

    # define evaluation method
    parser.add_argument('--evaluation_strategy', type=str, default="synflow",
                        help='which evaluation algorithm to use, support following options'
                             'grad_norm, synflow'
                             'grad_norm: '
                             'grad_plain: '
                             'jacob_conv: '
                             'nas_wot: '
                             'ntk_cond_num: '
                             'ntk_trace: '
                             'ntk_trace_approx: '
                             'fisher: '
                             'grasp: '
                             'snip: '
                             'synflow: '
                             'weight_norm: '
                             'use all of the above, one by one')

    # define device
    parser.add_argument('--device', type=str, default="cpu",
                        help='which device to use [cpu, cuda]')

    # search space configs for nasBench101
    parser.add_argument('--init_channels', default=128, type=int, help='output channels of stem convolution')
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')

    #  weight initialization settings, nasBench201 requires it.
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')

    parser.add_argument('--bn', type=int, default=1,
                        help="If use batch norm in network 1 = true, 0 = false")

    parser.add_argument('--arch_size', type=int, default=4,
                        help='How many node the architecture has at least')

    return parser.parse_args()


def vote():
    pass


# this will run in a separate server
def evaluate_arch(evaluator_dict, arch_id, architecture, used_search_space):

    final_score = 0
    final_cost = 0

    for evaluator_name, evaluator in evaluator_dict.items():
        new_arch = used_search_space.copy_architecture(arch_id, architecture)
        try:
            new_arch = new_arch.to(args.device)
            score, time_usage = evaluator.evaluate_wrapper(
                arch=new_arch,
                device=args.device,
                batch_data=mini_batch,
                batch_labels=mini_batch_targets)
            logger.info(evaluator_name + ":" + str(score))
            final_score += score
            final_cost += time_usage
        except Exception as e:
            if "out of memory" in str(e):
                logger.info("======================================================")
                logger.error("architecture " + str(arch_id) + " will be evaluate in CPU, message = " + str(e))
                logger.info("======================================================")
                mini_batch_cpu = mini_batch.cpu()
                mini_batch_targets_cpu = mini_batch_targets.cpu()
                new_arch_cpu = used_search_space.copy_architecture(arch_id, new_arch.cpu()).cpu()
                score, cpu_time_usage = evaluator.evaluate_wrapper(
                    arch=new_arch_cpu,
                    device="cpu",
                    batch_data=mini_batch_cpu,
                    batch_labels=mini_batch_targets_cpu)
                logger.info(evaluator_name + " on cpu:" + str(score))
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


if __name__ == '__main__':

    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    args = parse_arguments()

    if args.bn == 1:
        bn = True
    else:
        bn = False

    logger.info("running with params: :" + json.dumps(args.__dict__, indent=2))

    logger.info("cuda available = " + str(torch.cuda.is_available()))

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

    logger.info("1. Loaded batch data into GPU")
    # Show the utilization of all GPUs in a nice table
    showUtilization()

    used_search_space = search_space.init_search_space(args)

    # 1. Sampler one architecture
    sampler = sampler_register[args.arch_sampler](used_search_space, args)
    arch_generator = sampler.sample_next_arch(args.arch_size)

    # 2. Evaluator to evaluate one architecture
    evaluator_dict = {}
    metrics_usd = args.evaluation.split(", ")
    for ele in metrics_usd:
        metrics = ele.strip()
        evaluator_dict.update({ metrics: evaluator_register[metrics] })

    # only consider time
    total_time_cost = []
    k = 10

    try:
        iteration = 0
        history = {}
        # sampling process
        while len(total_time_cost) == 0 or sum(total_time_cost) < args.time_budget:
            iteration += 1
            print("Evaluate run with id =", iteration,
                  "cost =", sum(total_time_cost), "budget = ", args.time_budget)

            # sampling process
            arch_id, _ = arch_generator.__next__()
            architecture = used_search_space.new_architecture(arch_id)

            score_res, time_cost = evaluate_arch(evaluator_dict, arch_id, architecture, used_search_space)
            total_time_cost.append(time_cost)
            history[arch_id] = score_res

            sampler.fit_sampler(score_res)

        sorted_score = sorted(history.items(), key=lambda x: x[1], reverse=True)
        topK_arch_ids = [ele[0] for ele in sorted_score[-k:]]

        print("selected architecture = ", topK_arch_ids)

    except Exception as e:
        print(traceback.format_exc())
        logger.info("================================================================================================")
        logger.error(traceback.format_exc())
        logger.error("error: " + str(e))
        logger.info("================================================================================================")

