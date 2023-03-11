from eva_engine.phase2.algo.trainer import ModelTrainer
from utilslibs.parse_pre_res import SimulateTrain


class P2Evaluator:

    def __init__(self, search_space_name: str, dataset: str, is_simulate: bool = True):
        """
        :param search_space_name:
        :param dataset:
        :param is_simulate: train or not, default query from API.
        """
        self.space_name = search_space_name
        self.dataset = dataset
        self.is_simulate = is_simulate
        self.acc_getter = None

    def p2_evaluate(self, cand: str, epoch_per_model: int) -> float:
        """
        :param cand: candidate id
        :param epoch_per_model: epoch for each model
        :return:
        """
        if self.is_simulate:
            return self._evaluate_query(cand, epoch_per_model)
        else:
            return self._evaluate_train(cand, epoch_per_model)

    def _evaluate_query(self, cand: str, epoch_per_model: int) -> float:
        """
        :param cand: the candidate to evaluate
        :param epoch_per_model: how many resource it can use, epoch number
        :return:
        """
        if self.acc_getter is None:
            self.acc_getter = SimulateTrain(space_name=self.space_name,
                                            total_epoch=200)
        acc, _ = self.acc_getter.get_ground_truth(arch_id=cand, epoch_num=epoch_per_model, dataset=self.dataset)

        return acc

    def _evaluate_train(self, cand: str, epoch_per_model: int) -> float:
        """
        :param cand: the candidate to evaluate
        :param epoch_per_model: how many resource it can use, epoch number
        :return:
        """
        self.acc_getter = SimulateTrain(space_name=self.space_name,
                                        total_epoch=200)
        acc, _ = ModelTrainer.fully_train_arch(arch_id=cand, epoch_num=epoch_per_model, dataset=self.dataset)

        return acc
