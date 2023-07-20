# query ground truth
from src.common.constant import Config, CommonVars
from src.query_api.query_model_performance import Gt201, Gt101, GTMLP
from src.query_api.img_score import ImgScoreQueryApi
from typing import *


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
            self.api = Gt101()
            acc, time_usage = self.api.get_c10_test_info(arch_id, dataset, epoch_num)
            return acc, time_usage

        elif self.space_name == Config.NB201:
            self.api = Gt201()
            if total_epoch == 200:
                acc, time_usage = self.api.query_200_epoch(arch_id, dataset, epoch_num)
            else:  # 12
                acc, time_usage = self.api.query_12_epoch(arch_id, dataset, epoch_num)
            return acc, time_usage

        elif self.space_name == Config.MLPSP:
            self.api = GTMLP(dataset)
            acc, time_usage = self.api.get_valid_auc(arch_id, epoch_num)
            return acc, time_usage

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
            acc, _ = self.get_ground_truth(arch_id)
            if acc > cur_best:
                cur_best = acc
                res = arch_id
        return res

    def get_all_model_ids(self, dataset):
        if self.space_name == Config.NB101:
            self.api = Gt101()
        elif self.space_name == Config.NB201:
            self.api = Gt201()
        elif self.space_name == Config.MLPSP:
            self.api = GTMLP(dataset)
        return self.api.get_all_trained_model_ids()


class SimulateScore:
    def __init__(self, space_name: str, dataset_name: str):
        """
        :param space_name: NB101 or NB201, MLP
        :param dataset_name: NB101 or NB201, MLP
        """
        self.space_name = space_name
        if self.space_name == Config.MLPSP:
            self.api = GTMLP(dataset_name)
        else:
            self.api = ImgScoreQueryApi(self.space_name, dataset_name)

    # get the test_acc and time usage to train of this arch_id
    def get_score_res(self, arch_id) -> Dict:
        if self.space_name == Config.MLPSP:
            # todo: here we directly return the rank_score, instead of the mutilpel_algs score
            # return {"nas_wot": self.api.get_metrics_score(arch_id, dataset)["nas_wot"],
            #         "synflow": self.api.get_metrics_score(arch_id, dataset)["synflow"],
            #         }
            return self.api.get_global_rank_score(arch_id)
        else:
            return self.api.api_get_score(arch_id, CommonVars.PRUNE_SYNFLOW)

    def get_all_tfmem_score_res(self, arch_id) -> Dict:
        """
        return {alg_name: score}
        """
        return self.api.api_get_score(arch_id)

    def get_all_model_ids(self, dataset) -> List:
        """
        return all models_ids as a list
        """
        return self.api.get_all_scored_model_ids()
