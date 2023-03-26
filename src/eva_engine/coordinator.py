from common.constant import Config

from eva_engine.phase2.evaluator import P2Evaluator
from eva_engine.phase2.run_sh import SH
from logger import logger
from utilslibs.parse_pre_res import SimulateTrain

eta = 3


def schedule(dataset, sh: SH, T_,
             t1_, t2_,
             w_,
             search_space_ins, N_K_ratio, only_phase1: bool = False):
    """
    dataset: dataset name
    sh: sh
    T_: total time budget
    t1_: time for score a model
    t2_: time for train a model
    w_:  number of workers, only used in distributed EA
    space_name: search_space instance, 101, 201 MLP
    N_K_ratio: ratio
    only_phase1: if only using phase1.
    """
    if T_ < 1:
        raise

    # min time budget used for having both phases, when K = 1, N = min_budget_required_both_phase * 1
    if search_space_ins.name == Config.NB101:
        K_max = int(len(search_space_ins) / N_K_ratio)
        U_options = [4, 12, 16, 108]
        U_min = U_options[0]
        min_budget_required_both_phase = sh.pre_calculate_time_required(K=1, U=U_min)[1] + N_K_ratio * t1_
    elif search_space_ins.name == Config.NB201:
        K_max = int(len(search_space_ins) / N_K_ratio)
        U_options = list(range(1, 200))
        U_min = U_options[0]
        min_budget_required_both_phase = sh.pre_calculate_time_required(K=1, U=U_min)[1] + N_K_ratio * t1_
    elif search_space_ins.name == Config.MLPSP:
        K_max = int(len(search_space_ins) / N_K_ratio)
        # todo: this is for benchmark only
        if dataset == Config.Frappe:
            MaxEpochTrained = 20
        elif dataset == Config.Criteo:
            MaxEpochTrained = 10
        elif dataset == Config.UCIDataset:
            MaxEpochTrained = 40
        else:
            raise NotImplementedError
        U_options = list(range(1, MaxEpochTrained))

        U_min = U_options[0]
        min_budget_required_both_phase = sh.pre_calculate_time_required(K=1, U=U_min)[1] + N_K_ratio * t1_
    else:
        raise NotImplementedError

    history = []

    # if there is only phase 1
    enable_phase2_at_least = sh.pre_calculate_time_required(K=2, U=U_min)[1] + 2 * N_K_ratio * t1_
    if only_phase1 or enable_phase2_at_least > T_:
        # all time give to phase1
        for N_only in range(1, min(int(T_ / t1_), len(search_space_ins)) + 1):
            time_used = N_only * t1_
            if time_used > T_:
                break
            else:
                history.append((1, U_min, N_only))

        if len(history) == 0:
            logger.info(
                f" [FIRMEST] Only p1, Budget {T_} is too small, it's at least >= {time_used} "
                f"with current worker, {t1_}, {t2_}, eta")
            raise

    else:
        # record all possible K, U pair meeting the SLO ( time used < T)
        for K_ in range(2, min(int(T_ / t1_), K_max) + 1):
            N_ = K_ * N_K_ratio
            for U in U_options:
                # when K ==1, phase 2 is zero
                time_used = sh.pre_calculate_time_required(K=K_, U=U)[1] + N_ * t1_
                if time_used > T_:
                    break
                else:
                    history.append((K_, U, N_))

        if len(history) == 0:
            logger.info(
                f" [FIRMEST] Budget {T_} is too small, it's at least >= {min_budget_required_both_phase} "
                f"with current worker, {t1_}, {t2_}, eta")
            raise

    best_K, best_U, best_N = history[-1][0], history[-1][1], history[-1][2]
    N_scored = best_N
    B1_time_used = N_scored * t1_
    B2_all_epoch, B2_time_used = sh.pre_calculate_time_required(K=best_K, U=best_U)
    logger.info(
        f" [FIRMEST] The schedule result: when T = {T_} second, N = {N_scored}, K = {best_K}, best_U = {best_U}, time_used = {B1_time_used + B2_time_used}")
    return best_K, best_U, N_scored, B1_time_used, B2_time_used, B2_all_epoch
