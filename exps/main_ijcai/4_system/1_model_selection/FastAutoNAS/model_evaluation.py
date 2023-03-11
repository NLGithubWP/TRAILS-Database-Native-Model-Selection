

import json
import random
import time
import traceback
import numpy as np
import os
import torch
import argparse
import calendar

from controller.controler import Controller
base_dir = os.getcwd()


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--metrics_result', type=str,
                        default=os.path.join(base_dir, "result_base/result_system/prod/online_score_201_200run_3km_ea.json"),
                        help="metrics_result")

    # job config
    parser.add_argument('--num_run', type=int, default=5, help="num of run")
    parser.add_argument('--num_arch', type=int, default=20, help="how many architecture to evaluate")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default=os.path.join(base_dir,"data"),
                        help='path of data folder')

    # dataLoader setting,
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--num_data_workers', type=int, default=1, help='number of workers for dataLoader')
    parser.add_argument('--batch_sample_alg', type=str, default="random", help='sample a mini batch, [random, grasp]')

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench101",
                    help='search space to use, [nasbench101, nasbench201, ... ]')

    parser.add_argument('--api_loc', type=str, default="nasbench_only108.pkl",
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
    from eva_engine import evaluator_register
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
        sampler_new_t = []
        arch_gene_t = []
        arch_score_t = []
        sampler_fit = []
        total_worker_time = []
        y_axis_top10_models = []
        tfree_best = 0

        x_axis_time = []

        global_start_time = time.time()

        i = 1
        try:
            while True:
                if i > args.num_arch:
                    break

                # new arch
                sample_begin_time = time.time()
                arch_id, _ = arch_generator.__next__()
                arch_id_list.append(arch_id)
                sampler_end_time = time.time()
                sampler_new_t.append(sampler_end_time - sample_begin_time)

                # worker start from here
                model_eva_begin = time.time()

                # score NasWot
                newmodel_begin_time = time.time()
                new_arch = used_search_space.copy_architecture(arch_id, _, new_bn=True)
                new_arch = new_arch.to(args.device)
                newmodel_end_time = time.time()

                nw_begin_time = time.time()
                naswot_score, _ = evaluator_register[CommonVars.NAS_WOT].evaluate_wrapper(
                        arch=new_arch,
                        device=args.device,
                        batch_data=mini_batch,
                        batch_labels=mini_batch_targets)
                nw_end_time = time.time()

                # score SynFlow
                synflow_begin_time = time.time()
                new_arch = used_search_space.copy_architecture(arch_id, _, new_bn=False)
                new_arch = new_arch.to(args.device)
                synflow_end_time = time.time()
                arch_gene_t.append(synflow_end_time - synflow_begin_time + newmodel_end_time - newmodel_begin_time)

                synflow_begin_time = time.time()
                synflow_score, _ = evaluator_register[CommonVars.PRUNE_SYNFLOW].evaluate_wrapper(
                        arch=new_arch,
                        device=args.device,
                        batch_data=mini_batch,
                        batch_labels=mini_batch_targets)
                synflow_end_time = time.time()
                arch_score_t.append(synflow_end_time - synflow_begin_time + nw_end_time - nw_begin_time)

                # fit sampler
                alg_score = {CommonVars.NAS_WOT: naswot_score,
                             CommonVars.PRUNE_SYNFLOW: synflow_score}
                fit_begin_time = time.time()
                sampler.fit_sampler(arch_id, alg_score)
                fit_end_time = time.time()
                sampler_fit.append(fit_end_time - fit_begin_time)

                # clean old graph
                # del new_arch
                # torch.cuda.empty_cache()

                i = i + 1
                total_worker_time.append(time.time() - model_eva_begin)

                current_x_time = time.time() - global_start_time
                x_axis_time.append(current_x_time)

                # record arch_id with higher score
                top_arch_ids = sampler.get_current_top_k_models()
                y_axis_top10_models.append(top_arch_ids)

            all_run_info[run_id] = {}
            all_run_info[run_id]["arch_id_list"] = arch_id_list
            all_run_info[run_id]["sampler_new_t"] = sampler_new_t
            all_run_info[run_id]["arch_gene_t"] = arch_gene_t
            all_run_info[run_id]["arch_score_t"] = arch_score_t
            all_run_info[run_id]["sampler_fit"] = sampler_fit
            all_run_info[run_id]["total_worker_time"] = total_worker_time
            # record x and y information
            all_run_info[run_id]["y_axis_top10_models"] = y_axis_top10_models
            all_run_info[run_id]["x_axis_time"] = x_axis_time

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

