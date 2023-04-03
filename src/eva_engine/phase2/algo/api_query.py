

# query ground truth
from common.constant import Config
from query_api.query_model_gt_acc_api import Gt201, Gt101, GTMLP


class SimulateTrain:

    def __init__(self, space_name: str):
        """
        :param space_name: NB101 or NB201, MLP
        """
        self.space_name = space_name
        self.api = None

    # get the test_acc and time usage to train of this arch_id
    def get_ground_truth(self, arch_id, dataset=Config.c10, epoch_num=None, total_epoch: int = 200):
        """
        :param total_epoch: only 201 use it, 201 will query from 200 epoch in total.
        """
        if self.space_name == Config.NB101:
            if self.api is None:
                self.api = Gt101()
            score_, time_usage = self.api.get_c10_test_info(arch_id, dataset, epoch_num)
            return score_, time_usage

        elif self.space_name == Config.NB201:
            if self.api is None:
                self.api = Gt201()
            if total_epoch == 200:
                score_, time_usage = self.api.query_200_epoch(arch_id, dataset, epoch_num)
            else: # 12
                score_, time_usage = self.api.query_12_epoch(arch_id, dataset, epoch_num)
            return score_, time_usage

        elif self.space_name == Config.MLPSP:
            if self.api is None:
                self.api = GTMLP()
            score_, time_usage = self.api.get_valid_auc(arch_id, dataset, epoch_num)
            return score_, time_usage

        else:
            raise NotImplementedError

    # get the high acc of k arch with highest score
    def get_high_acc_top_10(self, top10):
        all_top10_acc = []
        time_usage = 0
        for arch_id in top10:
            score_, time_usage_ = self.get_ground_truth(arch_id)
            all_top10_acc.append(score_)
            time_usage += time_usage_
        return max(all_top10_acc), time_usage

    def get_best_arch_id(self, top10):
        cur_best = 0
        res = None
        for arch_id in top10:
            score_, _ = self.get_ground_truth(arch_id)
            if score_ > cur_best:
                cur_best = score_
                res = arch_id
        return res


class SimulateScore:
    def __init__(self, space_name: str):
        """
        :param space_name: NB101 or NB201, MLP
        """
        self.space_name = space_name
        self.api = None

    # get the test_acc and time usage to train of this arch_id
    def get_score_res(self, arch_id, dataset=Config.c10):
        if self.space_name == Config.MLPSP:
            if self.api is None:
                self.api = GTMLP()
            # todo: here we directly return the rank_score, instead of the mutilpel_algs score
            # return {"nas_wot": self.api.get_metrics_score(arch_id, dataset)["nas_wot"],
            #         "synflow": self.api.get_metrics_score(arch_id, dataset)["synflow"],
            #         }
            return self.api.get_global_rank_score(arch_id, dataset)
        else:
            raise NotImplementedError

