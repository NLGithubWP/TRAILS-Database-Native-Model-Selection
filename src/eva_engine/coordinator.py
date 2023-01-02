import math
import random

from common.constant import Config
from eva_engine.phase2.run_phase2 import P2Evaluator
from eva_engine.phase2.sh import SH
from query_api import gt_api
from query_api.parse_pre_res import FetchGroundTruth

eta = 3


def schedule(sh: SH, T_, t1_, t2_, w_, space_name, N_K_ratio, only_phase1: bool = False):
    if T_ < 1:
        raise
    # try different K and U combinations
    # only consider 15625 arches in this paper
    # min_budget_required: when K = 1, N = min_budget_required * 1
    if space_name == Config.NB101:
        K_max = int(15625 / N_K_ratio)
        U_options = [4, 12, 16, 108]
        U_min = U_options[0]
        min_budget_required = sh.pre_calculate_time_required(K=1, U=U_min)[1] + N_K_ratio * t1_
    else:
        K_max = int(15625 / N_K_ratio)
        U_options = list(range(1, 200))
        U_min = U_options[0]
        min_budget_required = sh.pre_calculate_time_required(K=1, U=U_min)[1] + N_K_ratio * t1_

    history = []

    # if there is only phase 1
    enable_phase2_at_least = sh.pre_calculate_time_required(K=2, U=U_min)[1] + 2 * N_K_ratio * t1_
    if only_phase1 == True or enable_phase2_at_least > T_:
        for N_only in range(1, min(int(T_ / t1_), 15625)):
            time_used = N_only * t1_
            if time_used > T_:
                break
            else:
                history.append((1, U_min, N_only))
    else:
        # record all possible K, U pair meeting the SLO ( time used < T)
        for K_ in range(2, min(int(T_ / t1_), K_max + 1)):
            N_ = K_ * N_K_ratio
            for U in U_options:
                # when K ==1, phase 2 is zero
                time_used = sh.pre_calculate_time_required(K=K_, U=U)[1] + N_ * t1_
                if time_used > T_:
                    break
                else:
                    history.append((K_, U, N_))

    # find the larges K and then large U
    if len(history) == 0:
        print(f"Budget {T_} is too small, it's at least >= {min_budget_required} with current worker, {t1_}, {t2_}, eta")
        raise
    else:
        best_K, best_U, best_N = history[-1][0], history[-1][1], history[-1][2]
        N_scored = best_N
        B1_time_used = N_scored * t1_
        B2_all_epoch, B2_time_used = sh.pre_calculate_time_required(K=best_K, U=best_U)
        print(f" The schedule result: when T = {T_} second, N = {N_scored}, K = {best_K}, best_U = {best_U}, time_used = {B1_time_used+B2_time_used}")
        return best_K, best_U, N_scored, B1_time_used, B2_time_used, B2_all_epoch


if __name__ == "__main__":
    W = 1
    N_K_ratio = 120

    budget_array = list(range(1, 320, 4))

    space_used = Config.NB101
    dataset_used = Config.c10

    for T in budget_array:
        Tsec = T*60
        if Tsec == 540:
            continue
        # print(f"T = {T}min, T = {Tsec} second")
        fgt = FetchGroundTruth(space_name=space_used, total_epoch=108)
        evaluator = P2Evaluator(fgt, dataset_used)

        t1 = gt_api.guess_score_time(space_used, dataset_used)
        time_per_epoch = gt_api.guess_train_one_epoch_time(space_used, dataset_used)

        sh = SH(evaluator, eta, time_per_epoch)

        best_K, best_U, N_scored, B1_time_used, B2_time_used = schedule(sh, Tsec, t1, time_per_epoch, W, space_used, N_K_ratio)
        # print(f"Budget={T}, real_time_used={B1_time_used + B2_time_used}, N={N_scored}, K={best_K}, U={best_U}")
