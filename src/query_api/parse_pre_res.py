import os

import numpy as np
from common.constant import Config
from query_api.gt_api import Gt201, Gt101
from utilslibs.tools import read_json


def gen_list_run_infos( data):
    result = []
    for run_id, value in data.items():
        res = EachRunInfo(run_id=run_id,
                          x_axis_time=data[run_id]["x_axis_time"],
                          y_axis_top10_model=data[run_id]["y_axis_top10_models"])
        result.append(res)
    return result


def get_current_best_acc(acc_list):
    res = []
    for ele in acc_list:
        if len(res) == 0:
            res.append(ele)
            continue
        if ele > res[-1]:
            res.append(ele)
        else:
            res.append(res[-1])
    return res


# query ground truth
class FetchGroundTruth:

    def __init__(self, space):
        self.space = space
        self.api = None

    # get the test_acc and time usage to train of this arch_id
    def get_ground_truth(self, arch_id, epoch_num=None, dataset=Config.c10):
        if self.space == Config.NB101:
            if self.api is None:
                self.api = Gt101()
            if epoch_num is None:
                epoch_num = 108
            score_, time_usage = self.api.get_c10_test_info(arch_id, epoch_num)
            return score_, time_usage

        if self.space == Config.NB201:
            if self.api is None:
                self.api = Gt201()
            if epoch_num is None:
                epoch_num = 199
            score_, time_usage = self.api.query_200_epoch(arch_id, Config.c10, epoch_num)
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


# accepting output of online system
# each run info, {x_list: time list, y: acc list,}
class EachRunInfo:
    def __init__(self, run_id, x_axis_time, y_axis_top10_model=None, y_current_best_accs=None):
        self.run_id = run_id
        self.x_axis_time = x_axis_time
        self.y_axis_top10_model = y_axis_top10_model
        self.y_current_best_accs = y_current_best_accs

    def get_current_best_acc(self, index, fgt: FetchGroundTruth):
        if self.y_current_best_accs is None:
            current_top_10 = self.y_axis_top10_model[index]
            high_acc, _ = fgt.get_high_acc_top_10(current_top_10)
            return high_acc
        else:
            return self.y_current_best_accs[index] * 0.01


# accepting output of online system
class ParseLatencyAll:
    def __init__(self, file_path: str, target, fgt: FetchGroundTruth, get_all_run_info):
        self.data = read_json(file_path)
        self.run_info = get_all_run_info(self.data)

        self.target = target
        self.fgt = fgt

    # measure the latency for each run
    def _get_latency_each_run(self, each_run: EachRunInfo, number_arch_to_target: dict):

        result = {}

        for i in range(1, len(each_run.x_axis_time)):
            # all target found
            if len(result) == len(self.target):
                break

            current_time = each_run.x_axis_time[i]
            high_acc = each_run.get_current_best_acc(i, self.fgt)

            for target_key, target_value in self.target.items():
                if target_key in result:
                    continue

                # record the number of models sampled before reaching the target.
                if target_key not in number_arch_to_target:
                    number_arch_to_target[target_key] = []

                if high_acc > target_value:
                    # 10 is server launch time.
                    result[target_key] = current_time
                    number_arch_to_target[target_key].append(i)

        return result

    # parse latency for each target over all runs, in form of {target_key: [50: a, 25: b, 75: c]...}
    def get_latency_quantile(self):
        number_arch_to_target = {}
        # {target1: [1,2,4...], target2: [1,2,34...]}
        target_time_list = {}
        for each_run in self.run_info:
            each_run_res = self._get_latency_each_run(each_run, number_arch_to_target)
            for target_key, latency in each_run_res.items():
                # record a list of time usage for each target
                if target_key not in target_time_list:
                    target_time_list[target_key] = []
                target_time_list[target_key].append(latency)

        quantile_latency = {}
        extra_info_model_num = []
        # now get quantile info
        for target_key, latency_list in target_time_list.items():
            # find all run with more than 1 models finding the target.
            num_arch_to_target_np = np.array([ele for ele in number_arch_to_target[target_key] if ele != 1])

            n_run_find_target = len(number_arch_to_target[target_key])
            num_arch_find_target25 = np.quantile(num_arch_to_target_np, 0.25, axis=0).item()
            num_arch_find_target5 = np.quantile(num_arch_to_target_np, 0.5, axis=0).item()
            num_arch_find_target75 = np.quantile(num_arch_to_target_np, 0.75, axis=0).item()

            latency_list_np = np.array(latency_list)
            mean25 = np.quantile(latency_list_np, .25, axis=0).item()
            mean5 = np.quantile(latency_list_np, .5, axis=0).item()
            mean75 = np.quantile(latency_list_np, .75, axis=0).item()

            quantile_latency[target_key] = {"mean25": mean25, "mean5": mean5, "mean75": mean75}

            extra_info_model_num.append([str(num_arch_find_target5), mean5])
            extra_info_model_num.append([str(num_arch_find_target25), mean25])
            extra_info_model_num.append([str(num_arch_find_target75), mean75])

        return quantile_latency, extra_info_model_num

    def get_throughput(self):
        all_tps = []
        total_models = 0
        for each_run in self.run_info:
            if len(each_run.x_axis_time) > 10:
                # measure without the first one.
                tps = len(each_run.x_axis_time) / each_run.x_axis_time[-1]
                all_tps.append(tps)
                total_models += len(each_run.x_axis_time)
            if total_models > 15000:
                break
        print(f"Measuring throughput with running total {total_models} models")
        mean5 = np.quantile(all_tps, .5, axis=0).item()
        return mean5


# accepting output of online system
def measure_time_usage(file_paths_list):
    total_t = []
    model_score_t = []
    model_gene_t = []

    for file in file_paths_list:
        data = read_json(file)

        total_t.extend(data["total_t"])
        model_score_t.extend(data["model_score_t"])
        model_gene_t.extend(data["model_gene_t"])

    print("Generating model using ", sum(model_gene_t) / sum(total_t))  # 0.533766836811139
    print("Scoring model using", sum(model_score_t) / sum(total_t))  # 0.46619504843604753

