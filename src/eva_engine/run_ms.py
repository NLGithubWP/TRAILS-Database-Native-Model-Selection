from eva_engine import coordinator
from eva_engine.phase1.run_phase1 import RunPhase1, run_phase1_simulate
from eva_engine.phase2.run_phase2 import P2Evaluator
from eva_engine.phase2.sh import SH
import search_space
import query_api.gt_api as gt_api


class RunModelSelection:

    def __init__(self, args, fgt, dataset, is_simulate: bool = False):
        eta = 3
        self.dataset = dataset
        self.args = args
        evaluator = P2Evaluator(fgt, dataset)
        self.time_per_epoch = gt_api.guess_train_one_epoch_time(self.args.search_space, self.dataset)
        self.sh = SH(evaluator, eta, self.time_per_epoch)
        self.is_simulate = is_simulate
        self.used_search_space = None

    def select_model(self, budget: float, run_id, only_phase1: bool = False, run_workers: int = 1):
        """
        :param only_phase1:
        :param budget: T in second
        :param run_id: run id,
        :param run_workers: how many workers
        :return:
        """

        # 0. profiling dataset and search space, get t1 and t2
        t1 = gt_api.guess_score_time(self.args.search_space, self.dataset)
        N_K_ratio = gt_api.profile_NK_trade_off(self.dataset)

        # 1. run coordinator to schedule
        K, U, N, B1_planed_time, B2_planed_time, B2_all_epoch = coordinator.schedule(self.sh, budget, t1,
                                                                                     self.time_per_epoch,
                                                                                     run_workers,
                                                                                     self.args.search_space,
                                                                                     N_K_ratio,
                                                                                     only_phase1)

        # 2. run phase 1 to score N models
        if self.is_simulate:
            K_models, B1_actual_time_use = run_phase1_simulate(self.args.search_space, self.dataset, run_id, N, K)
            # print("best model returned from Phase1 = ", K_models)
        else:
            # lazy loading the search space if needed.
            if self.used_search_space is None:
                self.used_search_space = search_space.init_search_space(self.args)
            # run phase-1 to get the K models.
            K_models, B1_actual_time_use = RunPhase1(self.args, K, N, self.used_search_space).run_phase1()

        # 3. run phase-2 to determine the final model
        best_arch, B2_actual_epoch_use = self.sh.run(U, K_models)
        # print("best model returned from Phase2 = ", K_models)

        return best_arch, B1_actual_time_use + B2_actual_epoch_use * self.time_per_epoch, B1_planed_time + B2_planed_time, B2_all_epoch
