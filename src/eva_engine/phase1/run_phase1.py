import json

from common.constant import Config
from common.structure import ModelAcquireData, ModelEvaData
from controller.controler import SampleController
from controller.sampler_all.seq_sampler import SequenceSampler

from eva_engine.phase1.evaluator import P1Evaluator
from logger import logger
from query_api.db_base import fetch_from_db
from query_api.query_model_gt_acc_api import Gt201, Gt101
from torch.utils.data import DataLoader
from controller import RegularizedEASampler
# Run ms in online, with scoring and updating controller.
from search_space.core.space import SpaceWrapper


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


class RunPhase1:

    def __init__(self, args, K: int, N: int, search_space_ins: SpaceWrapper,
                 train_loader: DataLoader, is_simulate: bool):
        """
        Each model selection job will init one class here.
        :param args: space, population_size, sample_size
        :param K: K models return in 1st phase
        :param N: N models eval in total
        :param search_space_ins:
        """

        # return K models
        self.K = K
        # explore N models
        self.N = N

        self.args = args
        if self.args.search_space == Config.NB201:
            self.gt_api = Gt201()
        elif self.args.search_space == Config.NB101:
            self.gt_api = Gt101()

        self.search_space_ins = search_space_ins

        # seq: init the search strategy and controller,

        if self.N == len(self.search_space_ins):
            print("Explore all models")
            strategy = SequenceSampler(self.search_space_ins)
        else:
            print("Explore with ea")
            strategy = RegularizedEASampler(self.search_space_ins,
                                            population_size=self.args.population_size,
                                            sample_size=self.args.sample_size)
        self.sampler = SampleController(strategy)

        # seq: init the phase 1 evaluator,
        self._evaluator = P1Evaluator(device=self.args.device,
                                      num_label=self.args.num_labels,
                                      dataset_name=self.args.dataset,
                                      search_space_ins=self.search_space_ins,
                                      train_loader=train_loader,
                                      is_simulate=is_simulate)

    def run_phase1(self) -> (list, list):
        """
        Controller explore n models, and return the top K models.
        :return:
        """
        current_models_perforamnces = []
        current_highest_score = []
        explored_n = 1
        model_eva = ModelEvaData()

        while explored_n <= self.N:
            # generate new model
            arch_id, arch_micro = self.sampler.sample_next_arch()
            # this is for sequence sampler.
            if arch_id is None:
                break
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
            logger.info("3. [trails] Phase 1: filter phase explored " + str(explored_n) +
                        " model, model_id = " + model_eva.model_id +
                        " model_scores = " + json.dumps(model_eva.model_score))

            self.sampler.fit_sampler(model_eva.model_id, model_eva.model_score, use_prue_score=self.args.use_prue_score)

            # this is to measure the value of metrix
            score_value = list(model_eva.model_score.values())[0] + list(model_eva.model_score.values())[1]
            if len(current_highest_score) == 0:
                current_highest_score.append(score_value)
                current_models_perforamnces.append(str(arch_id))
            else:
                if score_value > current_highest_score[-1]:
                    current_highest_score.append(score_value)
                    current_models_perforamnces.append(str(arch_id))
                else:
                    current_highest_score.append(current_highest_score[-1])
                    current_models_perforamnces.append(current_models_perforamnces[-1])

        # return the top K models
        return self.sampler.get_current_top_k_models(self.K), current_highest_score, current_models_perforamnces


