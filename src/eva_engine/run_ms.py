import time

from typing import Set, List

from eva_engine import coordinator
from eva_engine.phase1.run_phase1 import RunPhase1
from torch.utils.data import DataLoader
from eva_engine.phase2.run_sh import SH
from logger import logger
from search_space.init_search_space import init_search_space


class RunModelSelection:

    def __init__(self, search_space_name: str, dataset: str, args, is_simulate: bool = False):

        self.args = args

        self.eta = 3
        self.is_simulate = is_simulate
        # basic
        self.search_space_name = search_space_name
        self.dataset = dataset

        # p2 evaluator
        self.sh = None

        # instance of the search space.
        self.search_space_ins = init_search_space(self.args)

    def select_model_simulate(self, budget: float, run_id: int = 0, only_phase1: bool = False, run_workers: int = 1):
        """
        :param run_id:
        :param budget:  time budget
        :param only_phase1:
        :param run_workers:
        :return:
        """

        # 0. profiling dataset and search space, get t1 and t2

        score_time_per_model, train_time_per_epoch, N_K_ratio = self.search_space_ins.profiling(self.dataset)
        self.sh = SH(search_space_ins=self.search_space_ins,
                     dataset_name=self.dataset,
                     eta=self.eta,
                     time_per_epoch=train_time_per_epoch,
                     is_simulate=self.is_simulate)

        # 1. run coordinator to schedule
        K, U, N, B1_planed_time, B2_planed_time, B2_all_epoch = coordinator.schedule(self.dataset, self.sh, budget,
                                                                                     score_time_per_model,
                                                                                     train_time_per_epoch,
                                                                                     run_workers,
                                                                                     self.search_space_ins,
                                                                                     N_K_ratio,
                                                                                     only_phase1)

        # 2. run phase 1 to score N models
        K_models, B1_actual_time_use = RunPhase1.p1_evaluate_query(self.search_space_name, self.dataset, run_id, N, K)

        # 3. run phase-2 to determine the final model
        best_arch, B2_actual_epoch_use = self.sh.run_phase2(U, K_models)
        # print("best model returned from Phase2 = ", K_models)

        return best_arch, B1_actual_time_use + B2_actual_epoch_use * train_time_per_epoch, \
               B1_planed_time + B2_planed_time, B2_all_epoch

    def select_model_online(self, budget: float, data_loader: List[DataLoader],
                            only_phase1: bool = False, run_workers: int = 1):
        """
        Select model online
        :param budget:  time budget
        :param data_loader:  time budget
        :param only_phase1:
        :param run_workers:
        :return:
        """

        train_loader, valid_loader, test_loader = data_loader
        self.search_space_ins.load()

        logger.info("0. [FIRMEST] Begin model selection ... ")
        begin_time = time.time()

        logger.info("1. [FIRMEST] Begin profiling.")
        # 0. profiling dataset and search space, get t1 and t2
        score_time_per_model, train_time_per_epoch, N_K_ratio = self.search_space_ins.profiling(
            self.dataset,
            train_loader,
            valid_loader,
            self.args,
            is_simulate=self.is_simulate)

        self.sh = SH(search_space_ins=self.search_space_ins,
                     dataset_name=self.dataset,
                     eta=self.eta,
                     time_per_epoch=train_time_per_epoch,
                     is_simulate=self.is_simulate,
                     train_loader=train_loader,
                     val_loader=valid_loader,
                     args=self.args)

        # 1. run coordinator to schedule
        logger.info("2. [FIRMEST] Begin scheduling...")
        K, U, N, B1_planed_time, B2_planed_time, B2_all_epoch = coordinator.schedule(self.dataset, self.sh, budget,
                                                                                     score_time_per_model,
                                                                                     train_time_per_epoch,
                                                                                     run_workers,
                                                                                     self.search_space_ins,
                                                                                     N_K_ratio,
                                                                                     only_phase1)

        print(f"Budget = {budget}, N={N}, K={K}")

        # 2. run phase 1 to score N models
        logger.info("3. [FIRMEST] Begin to run phase1: filter phase")
        # lazy loading the search space if needed.

        # run phase-1 to get the K models.
        p1_runner = RunPhase1(
            args=self.args,
            K=K, N=N,
            search_space_ins=self.search_space_ins,
            train_loader=train_loader,
            is_simulate=self.is_simulate)

        K_models = p1_runner.run_phase1_seq()

        logger.info("4. [FIRMEST] Begin to run phase2: refinement phase")

        # 3. run phase-2 to determine the final model
        best_arch, best_arch_performance, B2_actual_epoch_use = self.sh.run_phase2(U, K_models)
        # print("best model returned from Phase2 = ", K_models)
        end_time = time.time()

        logger.info("5.  [FIRMEST] Real time Usage = " + str(end_time - begin_time)
                    + ", Final selected model = " + str(best_arch)
                    + ", planned time usage = " + str(B1_planed_time + B2_planed_time)
                    )

        return best_arch, best_arch_performance, end_time - begin_time, B1_planed_time + B2_planed_time, B2_all_epoch

    def schedule_only(self, budget: float, data_loader: List[DataLoader],
                            only_phase1: bool = False, run_workers: int = 1):
        """
        Select model online
        :param budget:  time budget
        :param data_loader:  time budget
        :param only_phase1:
        :param run_workers:
        :return:
        """

        train_loader, valid_loader, test_loader = data_loader
        self.search_space_ins.load()

        logger.info("0. [FIRMEST] Begin model selection ... ")
        begin_time = time.time()

        logger.info("1. [FIRMEST] Begin profiling.")
        # 0. profiling dataset and search space, get t1 and t2
        score_time_per_model, train_time_per_epoch, N_K_ratio = self.search_space_ins.profiling(
            self.dataset,
            train_loader,
            valid_loader,
            self.args,
            is_simulate=self.is_simulate)

        self.sh = SH(search_space_ins=self.search_space_ins,
                     dataset_name=self.dataset,
                     eta=self.eta,
                     time_per_epoch=train_time_per_epoch,
                     is_simulate=self.is_simulate,
                     train_loader=train_loader,
                     val_loader=valid_loader,
                     args=self.args)

        # 1. run coordinator to schedule
        logger.info("2. [FIRMEST] Begin scheduling...")
        K, U, N, B1_planed_time, B2_planed_time, B2_all_epoch = coordinator.schedule(self.dataset, self.sh, budget,
                                                                                     score_time_per_model,
                                                                                     train_time_per_epoch,
                                                                                     run_workers,
                                                                                     self.search_space_ins,
                                                                                     N_K_ratio,
                                                                                     only_phase1)

        return N


