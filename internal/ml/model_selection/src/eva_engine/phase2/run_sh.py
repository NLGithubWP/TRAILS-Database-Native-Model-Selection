
from copy import copy

from src.common.constant import Config
from src.eva_engine.phase2.evaluator import P2Evaluator

# successive halving
from src.logger import logger
from src.search_space.core.space import SpaceWrapper
from torch.utils.data import DataLoader


class BudgetAwareControllerSH:
    def __init__(self,
                 search_space_ins: SpaceWrapper, dataset_name: str,
                 eta, time_per_epoch,
                 train_loader: DataLoader = None,
                 val_loader: DataLoader = None,
                 args=None,
                 is_simulate: bool = True):
        """
        :param search_space_ins:
        :param dataset_name:
        :param time_per_epoch:
        :param is_simulate:
        :param eta: 1/mu to keep in each iteration
        """
        self._evaluator = P2Evaluator(search_space_ins, dataset_name,
                                      is_simulate=is_simulate,
                                      train_loader=train_loader, val_loader=val_loader,
                                      args=args)
        self.eta = eta
        self.max_unit_per_model = args.epoch
        self.time_per_epoch = time_per_epoch
        self.name = "SUCCHALF"

    def schedule_budget_per_model_based_on_T(self, space_name, fixed_time_budget, K_):
        # for benchmarking only phase 2

        # try different K and U combinations
        # only consider 15625 arches in this paper
        # min_budget_required: when K = 1, N = min_budget_required * 1
        if space_name == Config.NB101:
            U_options = [4, 12, 16, 108]
        else:
            U_options = list(range(1, 200))

        history = []

        for U in U_options:
            real_time_used = self.pre_calculate_epoch_required(K_, U) * self.time_per_epoch
            if real_time_used > fixed_time_budget:
                break
            else:
                history.append(U)
        if len(history) == 0:
            raise f"{fixed_time_budget} is too small for current config"
        return history[-1]

    def pre_calculate_time_required(self, K, U):
        all_epoch = self.pre_calculate_epoch_required(K, U)
        return all_epoch, all_epoch * self.time_per_epoch

    def pre_calculate_epoch_required(self, K, U):
        """
        :param K: candidates lists
        :param U: min resource each candidate needs
        :return:
        """
        total_epoch_each_rounds = K * U
        min_budget_required = 0
        previous_epoch = None
        while True:
            cur_cand_num = K
            if cur_cand_num == 1:
                break
            # number of each res given to each cand, pick lower bound
            epoch_per_model = int(total_epoch_each_rounds / cur_cand_num)
            if epoch_per_model >= self.max_unit_per_model:
                epoch_per_model = self.max_unit_per_model
            # evaluate each arch
            min_budget_required += epoch_per_model * cur_cand_num

            if previous_epoch is None:
                previous_epoch = epoch_per_model
            elif previous_epoch == epoch_per_model:
                # which means the epoch don't increase, no need to re-evaluate each component
                K = cur_cand_num - 1
                continue

            # sort from min to max
            if epoch_per_model == self.max_unit_per_model:
                # each model is fully evaluated, just return top 1
                K = 1
            else:
                # only keep 1/eta, pick lower bound
                num_keep = int(cur_cand_num * (1 / self.eta))
                if num_keep == 0:
                    num_keep = 1
                K = num_keep
        return min_budget_required

    def run_phase2(self, U: int, candidates_m: list) -> (str, float, float):
        """
        :param candidates_m: candidates lists
        :param U: min resource each candidate needs
        :return:
        """

        # print(f" *********** begin BudgetAwareControllerSH with U={U}, K={len(candidates_m)} ***********")
        candidates = copy(candidates_m)
        total_epoch_each_rounds = len(candidates) * U
        min_budget_required = 0
        previous_epoch = None
        scored_cand = None
        while True:
            total_score = []

            cur_cand_num = len(candidates)
            # number of each res given to each cand, pick lower bound
            epoch_per_model = int(total_epoch_each_rounds / cur_cand_num)

            logger.info("4. [trails] Phase 2: refinement phase, evaluating "
                        + str(cur_cand_num) + " models, with each using "
                        + str(epoch_per_model) + " epochs.")

            if cur_cand_num == 1:
                break

            if previous_epoch is None:
                previous_epoch = epoch_per_model
            elif previous_epoch == epoch_per_model:
                # which means the epoch don't increase, no need to re-evaluate each component
                num_keep = int(cur_cand_num * (1 / self.eta))
                if num_keep == 0:
                    num_keep = 1
                candidates = [ele[0] for ele in scored_cand[-num_keep:]]
                continue

            if epoch_per_model >= self.max_unit_per_model:
                epoch_per_model = self.max_unit_per_model
            # print(f"[run]: {cur_cand_num} model left, "
            #       f"and evaluate each model with {epoch_per_model} epoch")
            # evaluate each arch
            for cand in candidates:
                score, _ = self._evaluator.p2_evaluate(cand, epoch_per_model)
                total_score.append((cand, score))
                min_budget_required += epoch_per_model
            # sort from min to max
            scored_cand = sorted(total_score, key=lambda x: x[1])

            if epoch_per_model == self.max_unit_per_model:
                # each model is fully evaluated, just return top 1
                candidates = [scored_cand[-1][0]]
            else:
                # only keep 1/eta, pick lower bound
                num_keep = int(cur_cand_num * (1 / self.eta))
                if num_keep == 0:
                    num_keep = 1
                candidates = [ele[0] for ele in scored_cand[-num_keep:]]

        best_perform, _ = self._evaluator.p2_evaluate(candidates[0], self.max_unit_per_model)
        return candidates[0], best_perform, min_budget_required
