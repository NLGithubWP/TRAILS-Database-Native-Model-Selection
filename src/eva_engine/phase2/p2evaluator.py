from api_local.gt_api import Gt201, Gt101
from common.constant import Config


class P2Evaluator:

    def __init__(self, fgt):
        self.fgt = fgt

    def evaluate(self, cand: str, res_each_iter: int) -> float:
        """
        :param cand: the candidate to evalute
        :param res_each_iter: how many resource it can use, epoch number
        :return:
        """
        acc, _ = self.fgt.get_ground_truth(arch_id=cand, epoch_num=res_each_iter, dataset=Config.c10,)

        return acc






