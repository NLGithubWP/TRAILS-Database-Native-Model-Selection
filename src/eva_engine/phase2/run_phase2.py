
from common.constant import Config


class P2Evaluator:

    def __init__(self, fgt, simulate: bool = True):
        self.fgt = fgt
        self.simulate = simulate
        self.dataset = Config.c10

    def evaluate_query(self, cand: str, res_each_iter: int) -> float:
        """
        :param cand: the candidate to evalute
        :param res_each_iter: how many resource it can use, epoch number
        :return:
        """
        acc, _ = self.fgt.get_ground_truth(arch_id=cand, epoch_num=res_each_iter, dataset=self.dataset)

        return acc

    def evaluate_train(self, cand: str, res_each_iter: int) -> float:
        """
        :param cand: the candidate to evalute
        :param res_each_iter: how many resource it can use, epoch number
        :return:
        """
        pass

    def evaluate(self, cand: str, res_each_iter: int) -> float:
        if self.simulate:
            return self.evaluate_query(cand, res_each_iter)
        else:
            return self.evaluate_train(cand, res_each_iter)


