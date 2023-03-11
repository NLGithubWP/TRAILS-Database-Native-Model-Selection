

# query ground truth
from common.constant import Config
from query_api.query_model_gt_acc_api import Gt201, Gt101


class SimulateTrain:

    def __init__(self, space_name: str, total_epoch: int):
        """
        :param space_name: NB101 or NB201
        :param total_epoch: only 201 use it, 201 will query from 200 epoch in total.
        """
        self.space_name = space_name
        self.api = None
        self.total_epoch = total_epoch

    # get the test_acc and time usage to train of this arch_id
    def get_ground_truth(self, arch_id, dataset=Config.c10, epoch_num=None):
        if self.space_name == Config.NB101:
            if self.api is None:
                self.api = Gt101()
            score_, time_usage = self.api.get_c10_test_info(arch_id, dataset, epoch_num)
            return score_, time_usage

        if self.space_name == Config.NB201:
            if self.api is None:
                self.api = Gt201()
            if self.total_epoch == 200:
                score_, time_usage = self.api.query_200_epoch(arch_id, dataset, epoch_num)
            else: # 12
                score_, time_usage = self.api.query_12_epoch(arch_id, dataset, epoch_num)
            return score_, time_usage

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
