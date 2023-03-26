import json

from common.constant import Config
from common.structure import ModelAcquireData, ModelEvaData
from controller.controler import SampleController

from eva_engine.phase1.evaluator import P1Evaluator
from logger import logger
from query_api.db_base import fetch_from_db
from query_api.query_model_gt_acc_api import Gt201, Gt101
from torch.utils.data import DataLoader
from controller import RegularizedEASampler
# Run ms in online, with scoring and updating controller.
from search_space.core.space import SpaceWrapper


class RunPhase1:

    @staticmethod
    def p1_evaluate_query(space_name, dataset, run_id, N, K) -> (list, float):
        """
        :param space_name:
        :param dataset:
        :param run_id:
        :param N:
        :param K:
        :return: return list of models and time usage.
        """
        arch_id, candidates, current_time = fetch_from_db(space_name, dataset, run_id, N)
        return candidates[-K:], current_time

    def __init__(self, args, K: int, N: int, search_space_ins: SpaceWrapper, train_loader: DataLoader):
        """
        :param args: space, population_size, sample_size
        :param K: K models return in 1st phase
        :param N: N models eval in total
        :param search_space_ins:
        """

        self.args = args
        if self.args.search_space == Config.NB201:
            self.gt_api = Gt201()
        elif self.args.search_space == Config.NB101:
            self.gt_api = Gt101()

        self.search_space_ins = search_space_ins

        # seq: init the search strategy and controller,
        strategy = RegularizedEASampler(self.search_space_ins,
                                        population_size=self.args.population_size,
                                        sample_size=self.args.sample_size)

        self.sampler = SampleController(strategy)

        # seq: init the phase 1 evaluator,
        self._evaluator = P1Evaluator(device=self.args.device,
                                      num_label=self.args.num_labels,
                                      dataset_name=self.args.dataset,
                                      search_space_ins=self.search_space_ins,
                                      train_loader=train_loader)

        # return K models
        self.K = K
        # explore N models
        self.N = N

    def run_phase1_seq(self) -> list:
        """
        Controller explore n models, and return the top K models.
        :return:
        """

        explored_n = 0
        model_eva = ModelEvaData()

        while explored_n < self.N:
            if explored_n > 0:
                # fit sampler, None means first time acquire model
                self.sampler.fit_sampler(model_eva.model_id, model_eva.model_score, use_prue_score=False)

            # generate new model
            arch_id, arch_micro = self.sampler.sample_next_arch()
            model_encoding = self.search_space_ins.serialize_model_encoding(arch_micro)

            explored_n += 1

            # run the model selection
            model_acquire_data = ModelAcquireData(model_id=str(arch_id),
                                                  model_encoding=model_encoding,
                                                  is_last=False)
            data_str = model_acquire_data.serialize_model()

            # update the shared model eval res
            model_eva.model_id = str(arch_id)
            model_eva.model_score = self._evaluator.p1_evaluate(data_str)
            logger.info("3. [FIRMEST] Phase 1: filter phase explored " + str(explored_n) +
                        " model, model_id = " + model_eva.model_id +
                        " model_scores = " + json.dumps(model_eva.model_score))

        # return the top K models
        return self.sampler.get_current_top_k_models(self.K)


