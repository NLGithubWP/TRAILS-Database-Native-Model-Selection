from typing import List

from exps.main_v2.common.shared_args import parse_arguments
from utilslibs.compute import sample_in_log_scale_new
from utilslibs.draw_lib import draw_structure_data_anytime
from utilslibs.io_tools import read_json

args = parse_arguments()


def sample_some_points(x_array, y_2d_array, save_points, remove_n_points=1) -> (List, List):
    result_x_array = []
    result_y_array = []
    for run_id, time_list in enumerate(x_array):
        indices = sample_in_log_scale_new(time_list, save_points)
        # Sample the list using the calculated indices
        each_run_x_array = [time_list[i] for i in indices]
        each_run_y_array = [y_2d_array[run_id][int(i)] for i in indices]

        if remove_n_points != 0:
            result_x_array.append(each_run_x_array[:-remove_n_points])
            result_y_array.append(each_run_y_array[:-remove_n_points])
        else:
            result_x_array.append(each_run_x_array)
            result_y_array.append(each_run_y_array)

    return result_x_array, result_y_array


# 'frappe, criteo, uci_diabetes'

# dataset = "frappe"
# dataset = "criteo"
dataset = "uci_diabetes"


if dataset == "uci_diabetes":
    epoch = 0
    img_save_path = "./exps/main_v2/analysis/"
    sys_end2end_res = "./exps/main_v2/analysis/result/res_end_2_end_uci_diabetes_100_12_160k.json"
    sys_end2end_p1 = "./exps/main_v2/analysis/result/res_end_2_end_uci_diabetes_100_12_p1.json"
    mx_value = 67.4
    y_lim = [61.8, 67.5]
    figure_size = (6.2, 4.71)
    datasetfg_name = "Diabetes"
    annotations = [
        # ["TabNAS", 63.33, 8.14/60],
    ]
    remove_n_points=1
elif dataset == "frappe":
    epoch = 19
    img_save_path = "./exps/main_v2/analysis/"
    sys_end2end_res = "./exps/main_v2/analysis/result/res_end_2_end_frappe_100_12_15run.json"
    sys_end2end_p1 = "./exps/main_v2/analysis/result/res_end_2_end_frappe_100_12_p1.json"
    mx_value = 98.052
    y_lim = [97.6, None]
    figure_size = (6.2, 4.71)
    datasetfg_name=dataset
    annotations = [
        # ["TabNAS", 97.68, 324.8/60],
    ]
    remove_n_points = 2
elif dataset == "criteo":
    epoch = 9
    img_save_path = "./exps/main_v2/analysis/"
    sys_end2end_res = "./exps/main_v2/analysis/result/res_end_2_end_criteo_100_12.json"
    sys_end2end_p1 = "./exps/main_v2/analysis/result/res_end_2_end_criteo_100_12_p1.json"
    mx_value = 80.335
    y_lim = [80.1, None]
    figure_size = (6.2, 4.71)
    datasetfg_name=dataset
    annotations = [
        # ["TabNAS", 80.17, 7250.0/60],
    ]
    remove_n_points = 3
else:
    pass

# dataset = args.dataset
# epoch = args.epoch
# img_save_path = args.img_save_path
# sys_end2end_res = args.sys_end2end_res

print(f"reading from {sys_end2end_res}")

train_based_res = read_json(f"./exps/main_v2/analysis/result/res_train_base_line_{dataset}_epoch_{epoch}.json")
sampled_train_x, sampled_train_y = sample_some_points(x_array=train_based_res["baseline_time_budget"],
                                                      y_2d_array=train_based_res["baseline_acc"],
                                                      save_points=9,
                                                      remove_n_points=remove_n_points)

system_result = read_json(sys_end2end_res)
system_p1_result = read_json(sys_end2end_p1)

sampled_sys_x, sampled_sys_y = sample_some_points(x_array=[system_result["sys_time_budget"] for _ in system_result["sys_acc"]],
                                                  y_2d_array=system_result["sys_acc"],
                                                  save_points=7,
                                                  remove_n_points=0)


all_lines = [
    [sampled_train_x, sampled_train_y, "Training-Based MS"],
    [system_p1_result["sys_time_budget"], system_p1_result["sys_acc"], "Training-Free MS"],
    [sampled_sys_x, sampled_sys_y, "2Phase-MS"],
]

draw_structure_data_anytime(
    all_lines=all_lines,
    dataset=datasetfg_name,
    name_img=f"{img_save_path}/anytime_{dataset}",
    max_value=mx_value,
    figure_size=figure_size,
    annotations=annotations,
    y_ticks=y_lim,
    x_ticks=[0.01, None]
)

