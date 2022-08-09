

import json
import random
import time
import traceback
import numpy as np
import os
import torch
import argparse
import calendar

from sampler.controler import Controller


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--metrics_result', type=str, default="res_scoring.json", help="metrics_result")

    # job config
    parser.add_argument('--num_run', type=int, default=5, help="num of run")
    parser.add_argument('--num_arch', type=int, default=20, help="how many architecture to evaluate")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="/Users/kevin/project_python/Fast-AutoNAS/data",
                        help='path of data folder')

    # dataLoader setting,
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--num_data_workers', type=int, default=1, help='number of workers for dataLoader')
    parser.add_argument('--batch_sample_alg', type=str, default="random", help='sample a mini batch, [random, grasp]')

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench201",
                    help='search space to use, [nasbench101, nasbench201, ... ]')

    parser.add_argument('--api_loc', type=str, default="NAS-Bench-201-v1_0-e61699.pth",
                        help='which search space to use, ['
                             'nasbench101: nasbench_only108.pkl'
                             'nasbench201: NAS-Bench-201-v1_0-e61699.pth'
                             'nasbench201: NAS-Bench-201-v1_1-096897.pth'
                             ' ... ]')

    # define device
    parser.add_argument('--device', type=str, default="cpu",
                        help='which device to use [cpu, cuda]')

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
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_file_name", args.metrics_result[:-5]+"_"+str(ts)+".log" )

    from logger import logger
    from common.constant import CommonVars
    from storage import dataset
    from search_algorithm import evaluator_register
    import search_space

    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

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
        logger.info("batch_size is smaller than class_num " + str(args.batch_size) + " " + str(class_num) )
        # exit(0)

    # sample a batch with random or GRASP
    mini_batch, mini_batch_targets = dataset.get_mini_batch(
        dataloader=train_loader, sample_alg=args.batch_sample_alg, batch_size=args.batch_size, num_classes=class_num)

    mini_batch = mini_batch.to(args.device)
    mini_batch_targets = mini_batch_targets.to(args.device)

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
        current_x_time_list = []

        i = 1
        try:
            while True:
                if i > args.num_arch:
                    break

                # new arch
                arch_id, _ = arch_generator.__next__()
                arch_id_list.append(arch_id)





                # simulate training
                gt_dict = used_search_space.query_performance(int(arch_id), args.dataset)
                if gt_dict["test_accuracy"] > tfree_best:
                    tfree_best = gt_dict["test_accuracy"]

                current_best_acc.append(tfree_best)

                current_x_time_list.append(gt_dict["time_usage"])

                # fit sampler
                sampler.search_strategy.fit_sampler(gt_dict["test_accuracy"])
                i = i + 1

            all_run_info[run_id] = {}
            all_run_info[run_id]["arch_id_list"] = arch_id_list
            all_run_info[run_id]["current_best_acc"] = current_best_acc
            all_run_info[run_id]["current_x_time_list"] = current_x_time_list

        except Exception as e:
            logger.info("========================================================================")
            logger.error(traceback.format_exc())
            logger.error("error: " + str(e))
            logger.info("========================================================================")
            exit(1)

        print("run {} finished using {}".format(run_id, time.time() - run_begin_time))
        logger.info("run {} finished using {}".format(run_id, time.time() - run_begin_time))

    with open(args.metrics_result, 'w') as outfile:
        outfile.write(json.dumps(all_run_info))

