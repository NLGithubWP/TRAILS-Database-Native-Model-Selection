from common.constant import Config
from eva_engine.phase2.algo.trainer import ModelTrainer
from utilslibs.parse_pre_res import SimulateTrain
from search_space.core.space import SpaceWrapper
from torch.utils.data import DataLoader


class P2Evaluator:

    def __init__(self, search_space_ins: SpaceWrapper,
                 dataset: str,
                 is_simulate: bool = True,
                 train_loader: DataLoader = None,
                 val_loader: DataLoader = None,
                 args=None):
        """
        :param search_space_ins:
        :param dataset:
        :param is_simulate: train or not, default query from API.
        """
        self.search_space_ins = search_space_ins

        # dataset name
        self.dataset = dataset
        self.is_simulate = is_simulate
        self.acc_getter = None

        # for training only
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

    def p2_evaluate(self, cand: str, epoch_per_model: int) -> float:
        """
        :param cand: candidate id
        :param epoch_per_model: epoch for each model
        :return:
        """
        # if it's simulate or it's image dataset
        if self.is_simulate or self.search_space_ins.name in [Config.NB101, Config.NB201]:
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
            self.acc_getter = SimulateTrain(space_name=self.search_space_ins.name,
                                            total_epoch=200)
        acc, _ = self.acc_getter.get_ground_truth(arch_id=cand, epoch_num=epoch_per_model, dataset=self.dataset)

        return acc

    def _evaluate_train(self, cand: str, epoch_per_model: int) -> float:
        """
        :param cand: the candidate to evaluate
        :param epoch_per_model: how many resource it can use, epoch number
        :return:
        """
        acc, _ = ModelTrainer.fully_train_arch(search_space_ins=self.search_space_ins,
                                               arch_id=cand,
                                               dataset=self.dataset,
                                               epoch_num=epoch_per_model,
                                               train_loader=self.train_loader,
                                               val_loader=self.val_loader,
                                               args=self.args)

        return acc
