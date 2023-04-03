from typing import List

import numpy as np

from exps.main_sigmod.common.shared_args import parse_arguments
from utilslibs.draw_lib import draw_structure_data_anytime
from utilslibs.io_tools import read_json

args = parse_arguments()


def sample_few_points_from_fully_train(x_array, y_2d_array, remove_n_points=1) -> (List, List):

    # Set the log scale sampling factor
    factor = 10

    result_x_array = []
    result_y_array = []

    for run_id, time_list in enumerate(x_array):

        # Calculate the indices for log scale sampling
        start_exp = np.log10(1)
        end_exp = np.log10(len(time_list))
        indices = np.logspace(start_exp, end_exp, factor, endpoint=False, base=10).astype(int) - 1
        indices = np.unique(indices)

        # Sample the list using the calculated indices
        each_run_x_array = [time_list[i] for i in indices]
        each_run_y_array = [y_2d_array[run_id][int(i)] for i in indices]

        result_x_array.append(each_run_x_array[:-remove_n_points])
        result_y_array.append(each_run_y_array[:-remove_n_points])

    return result_x_array, result_y_array


# 'frappe, criteo, uci_diabetes'

dataset = "uci_diabetes"
epoch = 0
img_save_path = "./exps/main_sigmod/analysis/"
sys_end2end_res = "./exps/main_sigmod/analysis/result/res_end_2_end_uci_diabetes_100_12.json"

# dataset = "frappe"
# epoch = 0
# img_save_path = "./exps/main_sigmod/analysis/"
# sys_end2end_res = "./exps/main_sigmod/analysis/result/res_end_2_end_uci_diabetes_100_10.json"
#

# dataset = "criteo"
# epoch = 9
# img_save_path = "./exps/main_sigmod/analysis/"
# sys_end2end_res = "./exps/main_sigmod/analysis/result/res_end_2_end_criteo_100_12.json"


# dataset = args.dataset
# epoch = args.epoch
# img_save_path = args.img_save_path
# sys_end2end_res = args.sys_end2end_res

print(f"reading from {sys_end2end_res}")

train_result19 = read_json(f"./exps/main_sigmod/analysis/result/res_train_base_line_{dataset}_epoch_{epoch}.json")
sampled_train_x, sampled_train_y = sample_few_points_from_fully_train(x_array=train_result19["baseline_time_budget"],
                                                                      y_2d_array=train_result19["baseline_acc"],
                                                                      remove_n_points=1)

system_result = read_json(sys_end2end_res)


all_lines = [
    [sampled_train_x, sampled_train_y, "Train-Based-FullyTrain"],
    [system_result["sys_time_budget"], system_result["sys_acc"], "FIRMEST"],
]

# draw_structure_data_anytime(all_lines, "frappe", f"{args.sys_end2end_res[:-4]}")

draw_structure_data_anytime(
    all_lines=all_lines,
    dataset=dataset,
    name_img=f"{img_save_path}/anytime_{dataset}",
    y_ticks=None,
    x_ticks=None
)

