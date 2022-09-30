

import json
import random
import time

import numpy as np
import local_api
from draw import gather_all_run_result_vote, draw_graph_vote, gather_all_run_result
from controller import sampler_register
from logger import logger
import search_space
import torch
import argparse
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker

voteList = [['nas_wot', 'synflow']]
singleList = ["nas_wot", "synflow"]

dk = 10
d_sampler = "bohb"
d_is_vote = 0
if d_is_vote == 1:
    d_is_vote_str = "vote"
else:
    d_is_vote_str = "gt"


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--log_name', type=str, default="result.json", help="log_name")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="/Users/kevin/project_python/Fast-AutoNAS/data",
                        help='path of data folder')

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench201",
                        help='which search space to use, [nasbench101, nasbench201, ... ]')

    parser.add_argument('--api_loc', type=str, default="NAS-Bench-201-v1_0-e61699.pth",
                        help='which search space to use, ['
                             'nasbench101: nasbench_only108.pkl '
                             'nasbench201: NAS-Bench-201-v1_0-e61699.pth'
                             ' ... ]')

    # define device
    parser.add_argument('--device', type=str, default="cpu",
                        help='which device to use [cpu, cuda]')

    # search space configs for nasBench101
    parser.add_argument('--stem_out_channels', default=128, type=int, help='output channels of stem convolution')
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

    # define architecture sampler,
    parser.add_argument('--arch_sampler', type=str, default=d_sampler,
                        help='which architecture sampler to use, [test, random, rl, ea, BOHB ]')

    parser.add_argument('--total_run', type=int, default=10,
                        help='Total run number in benchmark stage. ')

    parser.add_argument('--num_arch_each_run', type=int, default=100,
                        help='How many arch to evaluate in each run')

    parser.add_argument('--pre_scored_data', type=str,
                        default="/Users/kevin/project_python/Fast-AutoNAS/result/CIFAR10_15625/vote_res/"
                                "201_15625_c10_128_unionBest.json",
                        help="log_name")

    parser.add_argument('--num_labels', type=int, default=10, help="class number ")
    parser.add_argument('--k', type=int, default=dk, help='How many node the architecture has at least')
    parser.add_argument('--img_name', type=str, default="{}_{}_top_{}.jpg".format(d_sampler, d_is_vote_str, dk), help='')
    parser.add_argument('--save_all_run_file', type=str, default="{}_{}_top_{}".format(d_sampler, d_is_vote_str, dk))

    # RL sampler's parameters,
    parser.add_argument('--rl_learning_rate', type=float, help="The learning rate for REINFORCE.", default=0.0001)
    parser.add_argument('--rl_EMA_momentum', type=float, default=0.9, help="The momentum value for EMA.")
    # EA sampler's parameters,
    parser.add_argument('--population_size', type=int, default=10, help="The learning rate for REINFORCE.")
    parser.add_argument('--sample_size', type=int, default=3, help="The momentum value for EMA.")

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
                 convert_func=None,
                 space=None,
                 dataBest=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.convert_func = convert_func

        self.space = space
        self.dataBest = dataBest

        self.acc_list = []
        self.ori_score_list = []
        self.arch_id_list = []

    def compute(self, config, budget, **kwargs):
        arch_struct = self.convert_func(config)
        arch_id = str(self.space.api.query_index_by_arch(arch_struct))
        score = float(dataBest[arch_id]["test_accuracy"])

        gt = local_api.api_get_ground_truth(arch_id, dataBest)

        self.ori_score_list.append(score)
        self.acc_list.append(gt)
        # print("worker evaluate {} architecture".format(len(self.acc_list)))
        self.arch_id_list.append(arch_id)

        return {"loss": 1-score, "info": arch_id}


def one_run_bohb(used_space, dataBest, alg_name):
    cs = used_space.get_topology_config_space()
    config2structure = used_space.config2topology_func()

    hb_run_id = "0"

    NS = hpns.NameServer(run_id=hb_run_id, host="localhost", port=0)
    ns_host, ns_port = NS.start()
    num_workers = 1

    workers = []
    for i in range(num_workers):
        w = MyWorker(
            nameserver=ns_host,
            nameserver_port=ns_port,
            convert_func=config2structure,
            space=used_space,
            dataBest=dataBest,
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


if __name__ == '__main__':

    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    args = parse_arguments()

    logger.info("running with params: :" + json.dumps(args.__dict__, indent=2))

    used_search_space = search_space.init_search_space(args)

    with open(args.pre_scored_data, 'r') as readfile:
        dataBest = json.load(readfile)

    all_run_result = {}
    for run_id in range(args.total_run):
        print("run id", run_id)
        run_begin_time = time.time()

        all_run_result[run_id] = {}
        # if this is vote
        alg_map = one_run_bohb(used_search_space, dataBest, "gt")
        all_run_result[run_id]["gt"] = alg_map["gt"]

        print("time takes = {}".format(time.time() - run_begin_time))

    with open(args.save_all_run_file, 'w') as outfile:
        outfile.write(json.dumps(all_run_result))

    # after all run, draw the graph
    simulate_system_performance, acc_all_run = gather_all_run_result(all_run_result, args.k, singleList)
    draw_graph_single(simulate_system_performance, acc_all_run, singleList)


