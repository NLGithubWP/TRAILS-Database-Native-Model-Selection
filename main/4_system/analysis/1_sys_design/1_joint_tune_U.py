

import random

from eva_engine.phase2.run_phase2 import P2Evaluator
from plot_libs.graph_lib import draw_grid_graph_with_budget
from query_api.db_ops import fetch_from_db
from query_api.parse_pre_res import FetchGroundTruth
from common.constant import Config
from eva_engine.phase2.sh import SH
import os
import numpy as np
import query_api.gt_api as gt_api


def B2_to_time(epochs):
    # in second
    p2_each_epoch_run_time = 20
    return epochs * p2_each_epoch_run_time


def time_to_B1(left_time):
    """
    Return how many time used, and how many models to be score
    :param left_time:
    :return:
    """
    # in second
    p1_score_time = random.randint(3315, 4502) * 0.0001
    B1_m = int(left_time/p1_score_time)
    if B1_m > 15624:
        B1_m = 15624
        real_time_usage = 15624 * p1_score_time
    else:
        real_time_usage = left_time
    return real_time_usage, B1_m


if __name__ == "__main__":
    base_dir = os.getcwd()

    used_space = Config.NB201
    used_dataset = Config.c10

    time_per_epoch = gt_api.guess_train_one_epoch_time(used_space, used_dataset)

    fgt = FetchGroundTruth(used_space, 200)
    evaluator = P2Evaluator(fgt, used_dataset)
    eta = 3
    sh = SH(evaluator, eta, time_per_epoch)

    y_k_array = [5, 10, 20, 50, 100]
    x_epoch_array = [1, 5, 10, 50]
    # budget_array = [0.1, 0.2, 0.3, 0.4, 0.5, 8, 16, 24, 32, 64, 128, 256]
    # mins
    # budget_array = [1, 5, 10, 15, 20, 30, 50, 100, 500, 1000]
    budget_array = [300]
    # budget_array = [180]

    # this is in hour
    for time_min in budget_array:
        T = time_min * 60

        two_D_run_acc = []
        two_D_run_BT = []
        two_D_run_b1 = []
        two_D_run_b2 = []

        B1_all = []
        B2_all = []

        # for each k, and each U, run 100 times
        for k in y_k_array:
            # each row is one run, each col is one res corresponding to one U
            each_run_k_acc = []
            each_run_k_bt = []
            each_b1_run = []
            each_b2_run = []
            for run_id in range(100):
                eac_k_acc = []
                eac_k_bt = []  # in mins
                each_b1 = []
                each_b2 = []
                for U in x_epoch_array:
                    B2_planed = sh.pre_calculate_epoch_required(k, U)
                    p2_time = B2_to_time(B2_planed)
                    if T - p2_time < 1:
                        eac_k_acc.append(-1)
                        eac_k_bt.append(-1)
                        each_b1.append(-1)
                        each_b2.append(-1)
                    else:
                        p1_time, B1 = time_to_B1(T - p2_time)  # how many to score in p1
                        arch_id, candidates_all, curr_time = fetch_from_db(used_space, used_dataset, run_id, B1)
                        best_arch, B2_used = sh.run(U, candidates_all[-k:])
                        assert B2_used == B2_planed
                        acc_sh_v, _ = fgt.get_ground_truth(best_arch)
                        # record for drawing
                        eac_k_acc.append(int(acc_sh_v * 10000) / 10000)
                        eac_k_bt.append((p1_time + p2_time) / 60)
                        each_b1.append(B1)
                        each_b2.append(B2_used)
                each_run_k_acc.append(eac_k_acc)
                each_run_k_bt.append(eac_k_bt)
                each_b1_run.append(each_b1)
                each_b2_run.append(each_b2)

            mean_acc = np.quantile(np.array(each_run_k_acc), .5, axis=0)
            mean_bt = np.quantile(np.array(each_run_k_bt), .5, axis=0)

            mean_b1 = np.quantile(np.array(each_b1_run), .5, axis=0)
            mean_b2 = np.quantile(np.array(each_b2_run), .5, axis=0)

            two_D_run_acc.append(mean_acc.tolist())
            two_D_run_BT.append(mean_bt.tolist())

            two_D_run_b1.append(mean_b1.tolist())
            two_D_run_b2.append(mean_b2.tolist())
        #
        # mask_lst = [[0, 3], [3, 1]]
        # for ele in mask_lst:
        #     x = ele[0]
        #     y = ele[1]
        #     two_D_run_acc[x][y] = -1
        #     two_D_run_BT[x][y] = -1
        #     two_D_run_b1[x][y] = -1
        #     two_D_run_b2[x][y] = -1

        draw_grid_graph_with_budget(
            two_D_run_acc, two_D_run_BT, two_D_run_b1, two_D_run_b2, f"{time_min}b1b2",
            y_k_array, x_epoch_array
        )

        print(f"T={T}, eta={eta}, Acc_list = ", two_D_run_acc)
        print(f"T={T}, eta={eta}, Budget_list = ", two_D_run_BT)
        print(f"T={T}, eta={eta}, two_D_run_b1 = ", two_D_run_b1)
        print(f"T={T}, eta={eta}, two_D_run_b2 = ", two_D_run_b2)
        print("\n")


