

import json
import random
import time
import numpy as np
from benchmark_sampling_online import get_score_result
from logger import logger
import search_space
import torch
import argparse
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from storage import dataset

voteList = [['nas_wot', 'synflow']]

# singleList = ["grad_norm", "nas_wot", "synflow", "snip"]
singleList = ["nas_wot", "synflow"]


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    parser.add_argument('--is_vote', type=int, default=1, help='is_vote or not')
    parser.add_argument('--out_folder', type=str,
                        default="./main/3_benchmark_sampler/101_cifar10_sampling_res/", help="data set name")

    parser.add_argument('--pre_scored_data', type=str,
                        default="./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json", help="1")

    parser.add_argument('--gt_file', type=str, default="./101_result", help="1")

    # job config
    parser.add_argument('--log_name', type=str, default="result.json", help="log_name")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="./data", help='path of data folder')

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

    # define architecture sampler,
    parser.add_argument('--arch_sampler', type=str, default="bohb",
                        help='which architecture sampler to use, [test, random, rl, ea, bohb ]')

    parser.add_argument('--total_run', type=int, default=300,
                        help='Total run number in benchmark stage. ')

    parser.add_argument('--num_arch_each_run', type=int, default=400,
                        help='How many arch to evaluate in each run')

    # bohb
    parser.add_argument(
        "--strategy",
        default="sampling",
        type=str,
        nargs="?",
        help="optimization strategy for the acquisition function",
    )
    parser.add_argument(
        "--min_bandwidth",
        default=0.3,
        type=float,
        nargs="?",
        help="minimum bandwidth for KDE",
    )
    parser.add_argument(
        "--num_samples",
        default=64,
        type=int,
        nargs="?",
        help="number of samples for the acquisition function",
    )
    parser.add_argument(
        "--random_fraction",
        default=0.33,
        type=float,
        nargs="?",
        help="fraction of random configurations",
    )
    parser.add_argument(
        "--bandwidth_factor",
        default=3,
        type=int,
        nargs="?",
        help="factor multiplied to the bandwidth",
    )
    return parser.parse_args()


class MyWorker(Worker):
    def __init__(self, *args,
                 space=None,
                 dataSet=None,
                 loapi=None,
                 alg_name=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.space = space
        self.dataSet = dataSet

        self.acc_list = []
        self.ori_score_list = []
        self.arch_id_list = []

        self.alg_name = alg_name

        self.loapi = loapi

    def compute(self, config, budget, **kwargs):
        arch_struct = self.space.config2arch_func(config)
        arch_id = str(self.space.arch_to_id(arch_struct))
        if arch_id != "-1":
            gt = loapi.api_get_ground_truth(arch_id, self.dataSet)
            score = get_score_result(arch_id, loapi, self.alg_name, self.space, args, mini_batch, mini_batch_targets)
            self.ori_score_list.append(score)
            self.acc_list.append(gt)
            self.arch_id_list.append(arch_id)
            rank_score = loapi.get_rank_score(self.ori_score_list)
        else:
            rank_score = 0

        return {"loss": 1-rank_score, "info": arch_id}


def one_run_bohb(used_space, alg_name, loapi):
    cs = used_space.get_configuration_space()

    hb_run_id = "0"

    NS = hpns.NameServer(run_id=hb_run_id, host="localhost", port=0)
    ns_host, ns_port = NS.start()
    num_workers = 1

    workers = []
    for i in range(num_workers):
        w = MyWorker(
            nameserver=ns_host,
            nameserver_port=ns_port,
            space=used_space,
            dataSet=args.dataset,
            loapi=loapi,
            alg_name=alg_name,
            run_id=hb_run_id,
            id=i,
        )
        w.run(background=True)
        workers.append(w)

    start_time = time.time()
    bohb = BOHB(
        configspace=cs,
        run_id=hb_run_id,
        eta=3,
        min_budget=1,
        max_budget=3,
        nameserver=ns_host,
        nameserver_port=ns_port,
        num_samples=args.num_samples,
        random_fraction=args.random_fraction,
        bandwidth_factor=args.bandwidth_factor,
        ping_interval=10,
        min_bandwidth=args.min_bandwidth,
    )

    results = bohb.run(args.num_arch_each_run, min_n_workers=num_workers)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    alg_map = {}
    alg_map[alg_name] = {}
    alg_map[alg_name]["ori_score"] = workers[0].ori_score_list
    alg_map[alg_name]["acc"] = workers[0].acc_list
    alg_map[alg_name]["arch_id"] = workers[0].arch_id_list

    return alg_map


class MyVoteWorker(Worker):
    def __init__(self, *args,
                 space=None,
                 dataSet=None,
                 loapi=None,
                 vote_com=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.space = space
        self.dataSet = dataSet

        self.alg_map = {}

        vote_comb_name = "_".join(vote_com)
        # init dict
        self.alg_map[vote_comb_name] = {}
        self.alg_map[vote_comb_name]["ori_score"] = []
        self.alg_map[vote_comb_name]["acc"] = []
        self.alg_map[vote_comb_name]["arch_id"] = []
        for alg_name in vote_com:
            self.alg_map[alg_name] = {}
            self.alg_map[alg_name]["ori_score"] = []

        self.vote_com = vote_com

        self.loapi = loapi

    def compute(self, config, budget, **kwargs):
        arch_struct = self.space.config2arch_func(config)
        arch_id = self.space.arch_to_id(arch_struct)
        if arch_id != "-1":
            gt = loapi.api_get_ground_truth(arch_id, self.dataSet)
            rank_score = 0.0
            # get final score with multiple votes.
            for alg_name in self.vote_com:
                score_ = get_score_result(arch_id, loapi, alg_name, self.space, args, mini_batch, mini_batch_targets)
                self.alg_map[alg_name]["ori_score"].append(score_)
                rank_ = loapi.get_rank_score(self.alg_map[alg_name]["ori_score"])
                rank_score += rank_

            # record acc, score
            self.alg_map[vote_comb_name]["acc"].append(gt)
            self.alg_map[vote_comb_name]["arch_id"].append(arch_id)
            self.alg_map[vote_comb_name]["ori_score"].append(rank_score)

            rank_score = loapi.get_rank_score(self.alg_map[vote_comb_name]["ori_score"])
        else:
            rank_score = 0
        return {"loss": 1-rank_score, "info": arch_id}


def one_run_bohb_vote(used_space, loapi, vote_com):
    cs = used_space.get_configuration_space()

    hb_run_id = "0"

    NS = hpns.NameServer(run_id=hb_run_id, host="localhost", port=0)
    ns_host, ns_port = NS.start()
    num_workers = 1

    workers = []
    for i in range(num_workers):
        w = MyVoteWorker(
            nameserver=ns_host,
            nameserver_port=ns_port,
            space=used_space,
            dataSet=args.dataset,
            loapi=loapi,
            vote_com=vote_com,
            run_id=hb_run_id,
            id=i,
        )
        w.run(background=True)
        workers.append(w)

    start_time = time.time()
    bohb = BOHB(
        configspace=cs,
        run_id=hb_run_id,
        eta=3,
        min_budget=1,
        max_budget=3,
        nameserver=ns_host,
        nameserver_port=ns_port,
        num_samples=args.num_samples,
        random_fraction=args.random_fraction,
        bandwidth_factor=args.bandwidth_factor,
        ping_interval=10,
        min_bandwidth=args.min_bandwidth,
    )

    results = bohb.run(args.num_arch_each_run, min_n_workers=num_workers)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    return workers[0].alg_map


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

    args.save_all_run_file = args.out_folder + "{}_{}_{}".format(args.arch_sampler, d_is_vote_str, args.dataset)
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
    loapi = local_api

    all_run_result = {}
    for run_id in range(args.total_run):
        print("run id", run_id)
        run_begin_time = time.time()

        all_run_result[run_id] = {}
        # if this is vote
        if args.is_vote == 1:
            for vote_com in voteList:
                vote_comb_name = "_".join(vote_com)
                alg_map = one_run_bohb_vote(used_search_space, loapi, vote_com)
                all_run_result[run_id][vote_comb_name] = alg_map[vote_comb_name]
        # this is not vote
        else:
            for alg_name in singleList:
                alg_map = one_run_bohb(used_search_space, alg_name, loapi)
                all_run_result[run_id][alg_name] = alg_map[alg_name]

        print("time takes = {}".format(time.time() - run_begin_time))

    # loapi.save_latest_data()

    with open(args.save_all_run_file, 'w') as outfile:
        outfile.write(json.dumps(all_run_result))

