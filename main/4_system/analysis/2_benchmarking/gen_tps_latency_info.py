import os

from query_api.parse_pre_res import ParseLatencyAll, FetchGroundTruth, gen_list_run_infos
from common.constant import Config


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
    fgt = FetchGroundTruth(space, 200)

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
