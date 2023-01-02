import math
from copy import copy
from random import randint
from common.constant import Config


# successive halving
class SH:
    def __init__(self, evaluator, eta, time_per_epoch, max_unit=200):
        """
        :param evaluator:
        :param eta: 1/mu to keep in each iteration
        :param max_unit:  for 201, it's 200, for 101 it's 108
        """
        self._evaluator = evaluator
        self.eta = eta
        self.max_unit_per_model = max_unit
        self.time_per_epoch = time_per_epoch
        self.name = "SUCCHALF"

    def schedule_budget_per_model_based_on_T(self, space_name, fixed_time_budget, K_):
        # for benchmarking only phase 2

        # try different K and U combinations
        # only consider 15625 arches in this paper
        # min_budget_required: when K = 1, N = min_budget_required * 1
        if space_name == Config.NB101:
            U_options = [4, 12, 16, 108]
        else:
            U_options = list(range(1, 200))

        history = []

        for U in U_options:
            real_time_used = self.pre_calculate_epoch_required(K_, U) * self.time_per_epoch
            if real_time_used > fixed_time_budget:
                break
            else:
                history.append(U)
        if len(history) == 0:
            raise f"{fixed_time_budget} is too small for current config"
        return history[-1]

    def pre_calculate_time_required(self, K, U):
        all_budget = self.pre_calculate_epoch_required(K, U)
        return all_budget, all_budget * self.time_per_epoch

    def pre_calculate_epoch_required(self, K, U):
        """
        :param K: candidates lists
        :param U: min resource each candidate needs
        :return:
        """
        total_epoch_each_rounds = K * U
        min_budget_required = 0
        previous_epoch = None
        while True:
            cur_cand_num = K
            if cur_cand_num == 1:
                break
            # number of each res given to each cand, pick lower bound
            epoch_per_model = int(total_epoch_each_rounds / cur_cand_num)
            if epoch_per_model >= self.max_unit_per_model:
                epoch_per_model = self.max_unit_per_model
            # evaluate each arch
            min_budget_required += epoch_per_model * cur_cand_num

            if previous_epoch is None:
                previous_epoch = epoch_per_model
            elif previous_epoch == epoch_per_model:
                # which means the epoch don't increase, no need to re-evaluate each component
                K = cur_cand_num - 1
                continue

            # sort from min to max
            if epoch_per_model == self.max_unit_per_model:
                # each model is fully evaluated, just return top 1
                K = 1
            else:
                # only keep 1/eta, pick lower bound
                num_keep = int(cur_cand_num * (1 / self.eta))
                if num_keep == 0:
                    num_keep = 1
                K = num_keep
        return min_budget_required

    def run(self, U: int, candidates_m: list):
        """
        :param candidates_m: candidates lists
        :param U: min resource each candidate needs
        :return:
        """

        # print(f" *********** begin SH with U={U}, K={len(candidates_m)} ***********")
        candidates = copy(candidates_m)
        total_epoch_each_rounds = len(candidates) * U
        min_budget_required = 0
        previous_epoch = None
        scored_cand = None
        while True:
            cur_cand_num = len(candidates)
            if cur_cand_num == 1:
                break
            total_score = []
            # number of each res given to each cand, pick lower bound
            epoch_per_model = int(total_epoch_each_rounds / cur_cand_num)

            if previous_epoch is None:
                previous_epoch = epoch_per_model
            elif previous_epoch == epoch_per_model:
                # which means the epoch don't increase, no need to re-evaluate each component
                num_keep = int(cur_cand_num * (1 / self.eta))
                if num_keep == 0:
                    num_keep = 1
                candidates = [ele[0] for ele in scored_cand[-num_keep:]]
                continue

            if epoch_per_model >= self.max_unit_per_model:
                epoch_per_model = self.max_unit_per_model
            # print(f"[run]: {cur_cand_num} model left, "
            #       f"and evaluate each model with {epoch_per_model} epoch")
            # evaluate each arch
            for cand in candidates:
                score = self._evaluator.evaluate(cand, epoch_per_model)
                total_score.append((cand, score))
                min_budget_required += epoch_per_model
            # sort from min to max
            scored_cand = sorted(total_score, key=lambda x: x[1])

            if epoch_per_model == self.max_unit_per_model:
                # each model is fully evaluated, just return top 1
                candidates = [scored_cand[-1][0]]
            else:
                # only keep 1/eta, pick lower bound
                num_keep = int(cur_cand_num * (1 / self.eta))
                if num_keep == 0:
                    num_keep = 1
                candidates = [ele[0] for ele in scored_cand[-num_keep:]]

        return candidates[0], min_budget_required


if __name__ == "__main__":
    from common.constant import Config
    from query_api.parse_pre_res import FetchGroundTruth
    from eva_engine.phase2.run_phase2 import P2Evaluator

    fgt = FetchGroundTruth(Config.NB201)
    evaluator = P2Evaluator(fgt, Config.c10)
    mu_ = 2
    sh = SH(evaluator, mu_, 200)
    K = 200
    U = 4
    candidates_ts = [randint(1, 15600) for i in range(K)]
    best_arch, min_budget_required = sh.run(U, candidates_ts)
    min_budget_required_planed = sh.pre_calculate_epoch_required(K, U)
    print(f"best arch id = {best_arch} and min-budget-used = {min_budget_required}, "
          f"min_budget_required_planed = {min_budget_required_planed}")
    acc_sh_v, _ = fgt.get_ground_truth(best_arch)
