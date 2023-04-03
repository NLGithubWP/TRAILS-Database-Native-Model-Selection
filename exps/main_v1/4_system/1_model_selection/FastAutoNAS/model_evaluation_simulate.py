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
from query_api.query_p1_score_api import LocalApi
import query_api.query_model_gt_acc_api as gt_api
base_dir = os.getcwd()


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--metrics_result',
                        type=str,
                        default=os.path.join(
                            base_dir,
                            "result_base/result_system/simulate/TFMEM_101_c10_100run_8k_models_score_sum"),
                        help="output folder")

    # job config
    parser.add_argument('--num_run', type=int, default=20, help="num of run")
    parser.add_argument('--num_arch', type=int, default=8000, help="how many architecture to evaluate")

    # dataLoader setting,
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100, ImageNet16-120]')

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench101", help='nasbench101, nasbench201')

    parser.add_argument('--api_loc', type=str, default="nasbench_only108.pkl",
                        help='which search space to use, ['
                             'nasbench101: nasbench_only108.pkl'
                             'nasbench201: NAS-Bench-201-v1_0-e61699.pth'
                             'nasbench201: NAS-Bench-201-v1_1-096897.pth'
                        )

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
    args.num_labels = 10
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_file_name", args.metrics_result[:-5] + "_ " + str(ts) + ".log")

    from logger import logger
    from common.constant import CommonVars, Config
    import search_space

    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)

    logger.info("running with params: :" + json.dumps(args.__dict__, indent=2))
    logger.info("cuda available = " + str(torch.cuda.is_available()))
    loapi = LocalApi(args.search_space, args.dataset)
    used_search_space = search_space.init_search_space(args, loapi)

    # we store the simulation data into sqlLite for quickly reading.
    import sqlite3
    tfmem_smt_file = args.metrics_result
    con = sqlite3.connect(tfmem_smt_file)
    try:
        con.execute("CREATE TABLE simulateExp(run_num, model_explored, cur_arch_id, top200_model_list, current_x_time)")
        con.execute("CREATE INDEX index_name on simulateExp (run_num, model_explored);")
    except:
        raise

    print("Begin all run ")
    all_run_info = {}
    for run_id in range(args.num_run):
        run_begin_time = time.time()
        # 1. Sampler one architecture
        controller = Controller(used_search_space, args)
        arch_generator = controller.sample_next_arch(args.arch_size)

        # for logging result
        arch_id_list = []
        y_axis_top10_models = []

        current_x_time = 0
        x_axis_time = []

        # record total time usage in reach run for debug time usage
        total_ge_time = 0
        total_fit_time = 0
        total_compute_time = 0
        total_record_time = 0

        i = 1
        try:
            while True:
                if i > args.num_arch:
                    break
                # new arch
                begin_ge_model = time.time()
                arch_id, _ = arch_generator.__next__()
                total_ge_time += time.time() - begin_ge_model

                # worker start from here
                # phase 1
                try:
                    begin_get_score = time.time()
                    naswot_score = loapi.api_get_score(str(arch_id), CommonVars.NAS_WOT)
                    synflow_score = loapi.api_get_score(str(arch_id), CommonVars.PRUNE_SYNFLOW)
                    total_compute_time += time.time() - begin_get_score
                    arch_id_list.append(arch_id)
                except:
                    continue

                # fit sampler
                begin_fit = time.time()
                alg_score = {CommonVars.NAS_WOT: naswot_score,
                             CommonVars.PRUNE_SYNFLOW: synflow_score}
                controller.fit_sampler(arch_id, alg_score, use_prue_score=True)
                total_fit_time += time.time() - begin_fit

                begin_record = time.time()
                current_x_time += gt_api.guess_score_time(args.search_space, args.dataset)
                x_axis_time.append(current_x_time)
                # record arch_id with higher score
                top_models = json.dumps(controller.get_current_top_k_models(400))
                y_axis_top10_models.append(top_models)

                insert_str = """
                    INSERT INTO simulateExp VALUES
                        ({}, {}, {}, "{}", {}) 
                """.format(run_id, i, arch_id, top_models, round(current_x_time, 4))

                con.execute(insert_str)
                total_record_time += time.time() - begin_record
                i = i + 1

            # record x and y information
            all_run_info[run_id] = {
                "arch_id_list": arch_id_list,
                "y_axis_top10_models": y_axis_top10_models,
                "x_axis_time": x_axis_time,
            }

        except Exception as e:
            print(traceback.format_exc())
            logger.info("========================================================================")
            logger.error(traceback.format_exc())
            logger.error("error: " + str(e))
            logger.info("========================================================================")
            exit(1)

        print("run {} finished using {}".format(run_id, time.time() - run_begin_time))
        print(f"total_ge_time = {total_ge_time}, total_fit_time = {total_fit_time}, total_compute_time = "
              f"{total_compute_time}, total_record_time = {total_record_time}")
        logger.info("run {} finished using {}".format(run_id, time.time() - run_begin_time))

    con.commit()
    # write_json(args.metrics_result, all_run_info)
