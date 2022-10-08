
from common.constant import Config
from eva_engine import coordinator
from eva_engine.phase1.run_phase1 import RunPhase1
from eva_engine.phase2.run_phase2 import P2Evaluator
from eva_engine.phase2.sh import SH
from query_api.parse_pre_res import FetchGroundTruth


class RunModelSelection:

    def __init__(self, used_search_space, args):
        self.args = args
        self.used_search_space = used_search_space
        self.run_id = 0

        self.fgt = FetchGroundTruth(Config.NB201)
        evaluator = P2Evaluator(self.fgt)
        eta = 3
        self.sh = SH(evaluator, eta)

    def select_model(self, budget: float, dataset: str, run_workers: int = 1):

        # 0.profiling dataset on search space, get t1 and t2
        t1 = self.used_search_space.guess_eval_time(dataset)
        t2 = 20 # this is on NB201+C10
        U = 1

        # system worker number
        # 1.run scheduler to get K
        K, N = coordinator.get_K(budget, t1, t2, run_workers)

        # run phase-1 to get the K models.
        K_models = RunPhase1(self.args, K, N).run_phase1_simulate(1)

        # run phase-2 to determ the final model
        best_arch = self.sh.SuccessiveHalving(1, K_models)
        acc_sh_v, _ = self.fgt.get_ground_truth(best_arch)

        return best_arch, acc_sh_v



