
# successive halving
import math
from random import randint


class SH:
    def __init__(self, evaluator, eta, max_unit=200):
        """
        :param evaluator:
        :param eta: 1/mu to keep in each iteration
        :param max_unit:  for 201, it's 200, for 101 it's 108
        """
        self._evaluator = evaluator
        self.eta = eta
        self.max_unit = max_unit

    def allocate_min_total_budget(self, n, r):
        total_rounds = math.log(n, self.eta)
        min_budget_required = int(total_rounds * n * r)
        if r >= self.max_unit or min_budget_required >= n * self.max_unit:
            total_rounds = 1
            # to control only one total round in sh process
            self.eta = n
            min_budget_required = n * self.max_unit

        # pick lower bound
        res_each_iter = n * r
        return total_rounds, min_budget_required, res_each_iter

    def SuccessiveHalving(self, r: int, candidates: list):
        """
        :param candidates: candidates lists
        :param B: total budget, total epoch number
        :param r: min resource each candidate needs
        :return:
        """
        n = len(candidates)
        _, min_budget_required, res_each_iter = self.allocate_min_total_budget(n, r)

        while True:
            cur_cand_num = len(candidates)
            if len(candidates) == 1:
                break
            total_score = []
            # number of each res given to each cand, pick lower bound
            res_each_cand = int(res_each_iter/cur_cand_num)
            # evaluate each arch
            for cand in candidates:
                score = self._evaluator.evaluate(cand, res_each_cand)
                total_score.append((cand, score))
            scored_cand = sorted(total_score, key=lambda x: x[1])

            # only keep 1/eta, pick lower bound
            num_keep = int(cur_cand_num*(1/self.eta))
            if num_keep == 0:
                num_keep = 1
            candidates = [ele[0] for ele in scored_cand[-num_keep:]]

        return candidates[0]


if __name__ == "__main__":
    from common.constant import Config
    from query_api.parse_pre_res import FetchGroundTruth
    from eva_engine.phase2.run_phase2 import P2Evaluator

    fgt = FetchGroundTruth(Config.NB201)
    evaluator = P2Evaluator(fgt)
    mu_ = 2
    sh = SH(evaluator, mu_)
    r = 200
    n = 200

    candidates = [randint(1, 15600) for i in range(n)]
    best_arch = sh.SuccessiveHalving(4, candidates)
    acc_sh_v, _ = fgt.get_ground_truth(best_arch)


