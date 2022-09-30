
# successive halving
import math


class SH:
    def __init__(self, evaluator):
        self._evaluator = evaluator

    @classmethod
    def allocate_min_total_budget(cls, n, mu, r):
        min_budget_required = math.log(n, mu) * n * r
        total_rounds = math.log(n, mu)
        res_each_iter = n * r
        return total_rounds, min_budget_required, res_each_iter

    def SuccessiveHalving(self, r: int, candidates: list, mu: int = 2):
        """
        :param candidates: candidates lists
        :param B: total budget, total epoch number
        :param r: min resource each candidate needs
        :param mu: 1/mu to keep in each iteration
        :return:
        """
        n = len(candidates)
        total_rounds, min_budget_required, res_each_iter = SH.allocate_min_total_budget(n, mu, r)

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

            # only keep 1/mu, pick lower bound
            num_keep = int(cur_cand_num*(1/mu))
            if num_keep == 0:
                num_keep = 1
            candidates = [ele[0] for ele in scored_cand[-num_keep:]]

        return candidates[0]


if __name__ == "__main__":
    print("sh")
