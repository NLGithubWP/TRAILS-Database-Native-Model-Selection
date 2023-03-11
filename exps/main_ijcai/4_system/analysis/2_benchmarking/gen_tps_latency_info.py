import os

import numpy as np
from eva_engine.phase2.algo.api_query import SimulateTrain
from utilslibs.parse_pre_res import gen_list_run_infos, EachRunInfo
from common.constant import Config

# accepting output of online system
from utilslibs.io_tools import read_json


class ParseLatencyAll:
    def __init__(self, file_path: str, target, fgt: SimulateTrain, get_all_run_info):
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

                if high_acc >= target_value:
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


def generate_draw_dict(all_r, target):
    mapper_dic = {0: 1, 1: 2, 2: 4, 3: 8}
    all_low = []
    all_high = []
    all_guesses_correct = []
    all_proportion_correct = []
    all_exp = []

    for id, r in enumerate(all_r):
        name = mapper_dic[id]
        for target_key in target:
            one_col = r[target_key]
            low_v = one_col["mean25"]
            high_v = one_col["mean75"]
            guesses_correct_e = target_key
            proportion_correct_e = one_col["mean5"]
            all_low.append(low_v)
            all_high.append(high_v)
            all_guesses_correct.append(guesses_correct_e)
            all_proportion_correct.append(proportion_correct_e)
            all_exp.append(name)

    res = {
        'exp': all_exp,
        'proportion_correct': all_proportion_correct,
        'guesses_correct': all_guesses_correct,
        'hdi_low': all_low,
        'hdi_high': all_high
    }

    return res


if __name__ == "__main__":
    base_dir = os.getcwd()

    space = Config.NB201
    fgt = SimulateTrain(space, 200)

    time_file_list = [
        os.path.join(base_dir, "result_base/result_system/prod/time_usage/4wk_time_usage_id2.res"),
    ]

    # measure_time_usage(time_file_list)
    # target = {"93.9%": 0.939, "93.7%": 0.937, "93.5%": 0.935, "93.3%": 0.933}
    target = {"93.3%": 93.3}

    # score_data_w1 = ParseLatencyAll(
    #     os.path.join(base_dir, "result_base/result_system/simulate/TFMEM_201_200run_3km_ea.json"),
    #     target, fgt, gen_list_run_infos)
    result_all_w2 = ParseLatencyAll(
        os.path.join(base_dir, "result_base/result_system/prod/latency_no_seed/2wk_500run_NB201_c10_all"),
        target, fgt, gen_list_run_infos)
    result_all_w4 = ParseLatencyAll(
        os.path.join(base_dir, "result_base/result_system/prod/latency_no_seed/4wk_500run_NB201_c10_all"),
        target, fgt, gen_list_run_infos)
    result_all_w8 = ParseLatencyAll(
        os.path.join(base_dir, "result_base/result_system/prod/latency_no_seed/8wk_500run_NB201_c10_all"),
        target, fgt, gen_list_run_infos)

    # r1, r1_points = score_data_w1.get_latency_quantile()
    r2, r2_points = result_all_w2.get_latency_quantile()
    r4, r4_points = result_all_w4.get_latency_quantile()
    r8, r8_points = result_all_w8.get_latency_quantile()

    draw_res_d = generate_draw_dict([r2, r4, r8], target)
    print(draw_res_d)

    # print(r1_points)
    print(r2_points)
    print(r4_points)
    print(r8_points)
    #
    # result_all_w12 = ParseLatencyAll(
    #     os.path.join(base_dir, "result_base/result_system/prod/throughput_seed/12wk_TPS_1run_NB201_c10_all"),
    #     target, fgt, gen_list_run_infos)
    # result_all_w14 = ParseLatencyAll(
    #     os.path.join(base_dir, "result_base/result_system/prod/throughput_seed/14wk_TPS_1run_NB201_c10_all"),
    #     target, fgt, gen_list_run_infos)
    # result_all_w16 = ParseLatencyAll(
    #     os.path.join(base_dir, "result_base/result_system/prod/throughput_seed/16wk_TPS_1run_NB201_c10_all"),
    #     target, fgt, gen_list_run_infos)
    #
    # print(score_data_w1.get_throughput())
    # print(result_all_w2.get_throughput())
    # print(result_all_w4.get_throughput())
    # print(result_all_w8.get_throughput())
    #
    # print(result_all_w12.get_throughput())
    # print(result_all_w14.get_throughput())
    # print(result_all_w16.get_throughput())
