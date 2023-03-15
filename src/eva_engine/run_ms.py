import time

from eva_engine import coordinator
from eva_engine.phase1.run_phase1 import RunPhase1
from torch.utils.data import DataLoader
import query_api.query_model_gt_acc_api as gt_api
from eva_engine.phase2.run_sh import SH
from logger import logger
from search_space.init_search_space import init_search_space


class RunModelSelection:

    def __init__(self, search_space_name, dataset, is_simulate: bool = False):

        eta = 3
        # basic
        self.search_space_name = search_space_name
        self.dataset = dataset

        # profiling result
        self.train_time_per_epoch = None
        self.score_time_per_model = None
        # this is confirmed by exp on NB201+c10
        self.N_K_ratio = None

        # p2 evaluator
        self.profiling()
        self.sh = SH(search_space_name, dataset, eta, self.train_time_per_epoch, is_simulate)

        # instance of the search space.
        self.search_space_ins = None

    def profiling(self):
        self.score_time_per_model = gt_api.guess_score_time(self.search_space_name, self.dataset)
        self.train_time_per_epoch = gt_api.guess_train_one_epoch_time(self.search_space_name, self.dataset)
        self.N_K_ratio = gt_api.profile_NK_trade_off(self.dataset)

    def select_model_simulate(self, budget: float, run_id: int = 0, only_phase1: bool = False, run_workers: int = 1):
        """
        :param run_id:
        :param budget:  time budget
        :param only_phase1:
        :param run_workers:
        :return:
        """

        # 0. profiling dataset and search space, get t1 and t2
        t1 = self.score_time_per_model
        t2 = self.train_time_per_epoch
        N_K_ratio = self.N_K_ratio

        # 1. run coordinator to schedule
        K, U, N, B1_planed_time, B2_planed_time, B2_all_epoch = coordinator.schedule(self.sh, budget,
                                                                                     t1,
                                                                                     t2,
                                                                                     run_workers,
                                                                                     self.search_space_name,
                                                                                     N_K_ratio,
                                                                                     only_phase1)

        # 2. run phase 1 to score N models
        K_models, B1_actual_time_use = RunPhase1.p1_evaluate_query(self.search_space_name, self.dataset, run_id, N, K)

        # 3. run phase-2 to determine the final model
        best_arch, B2_actual_epoch_use = self.sh.run(U, K_models)
        # print("best model returned from Phase2 = ", K_models)

        return best_arch, B1_actual_time_use + B2_actual_epoch_use * t2, B1_planed_time + B2_planed_time, B2_all_epoch

    def select_model_online(self, budget: float, train_loader: DataLoader,
                            args=None, only_phase1: bool = False, run_workers: int = 1):
        """
        :param train_loader:
        :param budget:  time budget
        :param args:
        :param only_phase1:
        :param run_workers:
        :return:
        """

        if self.search_space_ins is None:
            self.search_space_ins = init_search_space(args)

        logger.info("0. [FIRMEST] Begin model selection ... ")
        begin_time = time.time()

        logger.info("1. [FIRMEST] Begin profiling.")
        # 0. profiling dataset and search space, get t1 and t2
        t1 = self.score_time_per_model
        t2 = self.train_time_per_epoch
        N_K_ratio = self.N_K_ratio

        # 1. run coordinator to schedule
        logger.info("2. [FIRMEST] Begin scheduling...")
        K, U, N, B1_planed_time, B2_planed_time, B2_all_epoch = coordinator.schedule(self.sh, budget,
                                                                                     t1,
                                                                                     t2,
                                                                                     run_workers,
                                                                                     self.search_space_name,
                                                                                     N_K_ratio,
                                                                                     only_phase1)

        # 2. run phase 1 to score N models
        logger.info("3. [FIRMEST] Begin to run phase1: filter phase")
        # lazy loading the search space if needed.

        # run phase-1 to get the K models.
        p1_runner = RunPhase1(args, K, N, self.search_space_ins, train_loader)
        K_models = p1_runner.run_phase1_seq()

        logger.info("4. [FIRMEST] Begin to run phase2: refinement phase")

        # 3. run phase-2 to determine the final model
        best_arch, B2_actual_epoch_use = self.sh.run(U, K_models)
        # print("best model returned from Phase2 = ", K_models)
        end_time = time.time()

        logger.info("5.  [FIRMEST] Real time Usage = " + str(end_time - begin_time)
                    + ", Final selected model = " + str(best_arch)
                    + ", planned time usage = " + str(B1_planed_time + B2_planed_time)
                    )

        return best_arch, end_time - begin_time, B1_planed_time + B2_planed_time, B2_all_epoch
