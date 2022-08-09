import json
import os
from scipy import stats


def read_file(score_file_path):
    with open(score_file_path, 'r') as readfile:
        data = json.load(readfile)
    all_metrics = ['grad_norm', 'grad_plain', 'jacob_conv', 'nas_wot', 'ntk_cond_num', 'ntk_trace', 'ntk_trace_approx', 'fisher', 'grasp', 'snip', 'synflow', 'weight_norm']
    for arch_id, info in data.items():
        print("arch_id = ", arch_id)
        for metrics_name, score_str in info.items():
            score_float = float(score_str)
            print(metrics_name, score_float)
        break


def measure_correlation(x1: list, x2: list):
    correlation1, p_value = stats.kendalltau(x1, x2, nan_policy='omit')
    correlation2, p_value = stats.spearmanr(x1, x2, nan_policy='omit')
    correlation3, p_value = stats.pearsonr(x1, x2)
    print("kendalltau", correlation1)
    print("pearsonr", correlation3)
    print("spearmanr", correlation2)


def read_grond_truth(gt_file_path):
    with open(gt_file_path, 'r') as readfile:
        gt = json.load(readfile)

    for arch_id_str, info in gt.items():
        print("arch_id",  arch_id_str)
        print("test-acc", float(info["cifar10"]["test-accuracy"]))
        break


if __name__ == "__main__":
    base_dir = "./"
    file_name = "201_15625_c10_bs32_ic16_unionBest.json"
    gt_file_name = "201_200_result"

    output_dir = os.path.join(base_dir, file_name)
    gt_file_path = os.path.join(base_dir, gt_file_name)

    read_grond_truth(gt_file_path)
    read_file(output_dir)

    x1 = [1,2,3]
    x2 = [4,5,6]
    measure_correlation(x1, x2)







