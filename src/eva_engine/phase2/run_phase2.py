

class P2Evaluator:

    def __init__(self, fgt, dataset):
        self.fgt = fgt
        self.dataset = dataset

    def evaluate_query(self, cand: str, epoch_per_model: int) -> float:
        """
        :param cand: the candidate to evaluate
        :param epoch_per_model: how many resource it can use, epoch number
        :return:
        """
        acc, _ = self.fgt.get_ground_truth(arch_id=cand, epoch_num=epoch_per_model, dataset=self.dataset)

        return acc

    def evaluate_train(self, cand: str, epoch_per_model: int) -> float:
        """
        :param cand: the candidate to evalute
        :param epoch_per_model: how many resource it can use, epoch number
        :return:
        """
        pass

    def evaluate(self, cand: str, epoch_per_model: int, simulate: bool = True) -> float:
        """
        :param simulate: simulate
        :param cand: candidate id
        :param epoch_per_model: epoch for each model
        :return:
        """
        if simulate:
            return self.evaluate_query(cand, epoch_per_model)
        else:
            return self.evaluate_train(cand, epoch_per_model)


