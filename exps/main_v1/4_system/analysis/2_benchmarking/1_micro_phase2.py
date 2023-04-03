import argparse
import math
import os
import time

import numpy as np
from matplotlib import pyplot as plt

import random
import matplotlib

frontsizeall = 20
marksizeall = 30

# points' mark size
set_tick_size = 15
# update tick size
matplotlib.rc('xtick', labelsize=set_tick_size)
matplotlib.rc('ytick', labelsize=set_tick_size)

plt.rcParams['axes.labelsize'] = set_tick_size


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')
    parser.add_argument('--epoch', type=int, default=200)
    return parser.parse_args()


def run_one_fixed_budget_alg(sh, time_per_epoch):

    # calculate min time required for evaluating 500 models
    min_epoch_for_fixed_k = sh.pre_calculate_epoch_required(total_models, 1)
    min_time_for_fixed_k = math.ceil(time_per_epoch * min_epoch_for_fixed_k)

    if sh.name == "SUCCREJCT":
        fully_train_each_model = int(244336900/500 * total_models)
        step = int((fully_train_each_model - min_time_for_fixed_k) / 30)
    else:
        fully_train_each_model = math.floor(total_models * 200 * time_per_epoch)
        step = int((fully_train_each_model - min_time_for_fixed_k) / 30)

    acc_reached = []
    time_used = []
    
    for run_id in range(total_run):
        begin_time = time.time()
        acc_each_run = []
        time_each_run = []
        for time_budget_used in range(min_time_for_fixed_k, fully_train_each_model, step):
            begin_time_u = time.time()
            U = sh.schedule_budget_per_model_based_on_T(search_space, time_budget_used, total_models)
            end_time_u = time.time()
            # print(f"run_id = {run_id}, time_usage for U = {end_time_u - begin_time_u}")

            begin_time_u = time.time()
            best_arch, _, B2_actual_epoch_use = sh.run_phase2(U, all_models[run_id])
            end_time_u = time.time()
            # print(f"run_id = {run_id}, time_usage for run = {end_time_u - begin_time_u}")

            begin_time_u = time.time()
            acc_sh_v, _ = fgt.get_ground_truth(arch_id=best_arch, dataset=dataset, epoch_num=None)
            end_time_u = time.time()
            # print(f"run_id = {run_id}, get ground truth for run = {end_time_u - begin_time_u}")

            acc_each_run.append(acc_sh_v)
            time_each_run.append(B2_actual_epoch_use/60)
            print(
                f" *********** begin with U={U}, K={len(all_models[run_id])}, B2_actual_epoch_use = {B2_actual_epoch_use}, acc = {acc_sh_v}, fully_train_each_model = {fully_train_each_model}, ***********")
        end_time = time.time()
        print(f"run_id = {run_id}, time_usage = {end_time-begin_time}")

        acc_reached.append(acc_each_run)
        time_used.append(time_each_run)
    
    # time_used = np.array(time_used)/60
    # acc_reached = np.array(acc_reached)
    # time_used_mean = np.quantile(time_used, .5, axis=0)
    #
    # accuracy_exp = np.array(acc_reached)
    # accuracy_q_75 = np.quantile(accuracy_exp, .75, axis=0)
    # accuracy_q_25 = np.quantile(accuracy_exp, .25, axis=0)
    # accuracy_mean = np.quantile(accuracy_exp, .5, axis=0)
    # # plot accuracy
    # plt.plot(time_used_mean, accuracy_mean, "o-", label=sh.name)
    # plt.fill_between(time_used_mean, accuracy_q_25, accuracy_q_75, alpha=0.1)

    return acc_reached, time_used


if __name__ == "__main__":

    os.environ.setdefault("base_dir", "../exp_data/")
    import query_api.query_model_gt_acc_api as gt_api

    from eva_engine.phase2.algo.api_query import SimulateTrain
    from eva_engine.phase2.evaluator import P2Evaluator
    from eva_engine.phase2.run_sh import BudgetAwareControllerSH
    from eva_engine.phase2.run_sr import SR
    from eva_engine.phase2.run_uniform import UniformAllocation
    from search_space.nas_201_api.model_params import NB201MacroCfg
    from search_space.nas_201_api.space import NasBench201Space
    from utilslibs.io_tools import write_json


    args = parse_arguments()
    total_run = 50
    random.seed(560)
    total_models = 500

    # pre_generate 100 * 500 models,
    all_models = []
    for run_id in range(total_run):
        _models = random.sample(list(range(1, 15624)), total_models)
        all_models.append(_models)

    search_space = "nasbench201"
    # dataset = "cifar10"
    # dataset = "cifar100"
    dataset = "ImageNet16-120"

    model_cfg = NB201MacroCfg(None, None,None, None, None)
    space_ins = NasBench201Space(None, model_cfg)
    time_per_epoch = gt_api.guess_train_one_epoch_time(search_space, dataset)
    fgt = SimulateTrain(space_name=search_space)
    evaluator = P2Evaluator(space_ins, dataset)

    result_save_dic = {}

    print("--- benchmarking sh_")
    sh_ = BudgetAwareControllerSH(search_space_ins=space_ins,
                                  dataset_name=dataset,
                                  eta=3, time_per_epoch=time_per_epoch,
                                  args=args)
    acc_reached, time_used = run_one_fixed_budget_alg(sh_, time_per_epoch)
    result_save_dic["sh"] = {
        "time_used": time_used,
        "acc_reached": acc_reached,
    }

    print("--- benchmarking uniform_")
    uniform_ = UniformAllocation(evaluator, time_per_epoch)
    acc_reached, time_used = run_one_fixed_budget_alg(uniform_, time_per_epoch)

    result_save_dic["uniform"] = {
        "time_used": time_used,
        "acc_reached": acc_reached,
    }

    print("--- benchmarking sr_")
    sr_ = SR(evaluator, time_per_epoch)
    acc_reached, time_used = run_one_fixed_budget_alg(sr_, time_per_epoch)

    result_save_dic["sr"] = {
        "time_used": time_used,
        "acc_reached": acc_reached,
    }

    write_json(f"micro_phase2_{dataset}", result_save_dic)
