from typing import List

from src.tools.compute import sample_in_log_scale_new
from src.tools.io_tools import read_json
from exps.draw_tab_lib import draw_structure_data_anytime
import os
import numpy as np
from pprint import pprint


def get_dataset_parameters(dataset):
    parameters = {
        "frappe": {
            "epoch": 19,
            "express_flow": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_express_flow.json",
            "express_flow_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_express_flow_p1.json",
            "fisher": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_fisher.json",
            "fisher_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_fisher_p1.json",
            "grad_norm": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_grad_norm.json",
            "grad_norm_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_grad_norm_p1.json",
            "grasp": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_grasp.json",
            "grasp_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_grasp_p1.json",
            "nas_wot": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_nas_wot.json",
            "nas_wot_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_nas_wot_p1.json",
            "ntk_cond_num": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_ntk_cond_num.json",
            "ntk_cond_num_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_ntk_cond_num_p1.json",
            "ntk_trace": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_ntk_trace.json",
            "ntk_trace_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_ntk_trace_p1.json",
            "ntk_trace_approx": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_ntk_trace_approx.json",
            "ntk_trace_approx_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_ntk_trace_approx_p1.json",
            "snip": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_snip.json",
            "snip_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_snip_p1.json",
            "synflow": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_synflow.json",
            "synflow_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_synflow_p1.json",
            "mx_value": 98.099,
            "y_lim": [None, None],
            "figure_size": (6.2, 4.71),
            "datasetfg_name": dataset,
            "annotations": [],  # ["TabNAS", 97.68, 324.8/60],
            "remove_n_points": 2,
        },
    }

    return parameters.get(dataset, None)


def sample_some_points(x_array, y_2d_array, save_points, remove_n_points=1) -> (List, List):
    result_x_array = []
    result_y_array = []
    for run_id, time_list in enumerate(x_array):
        indices = sample_in_log_scale_new(time_list, save_points)
        each_run_x_array = [time_list[i] for i in indices]
        each_run_y_array = [y_2d_array[run_id][int(i)] for i in indices]

        result_x_array.append(each_run_x_array[:-remove_n_points] if remove_n_points != 0 else each_run_x_array)
        result_y_array.append(each_run_y_array[:-remove_n_points] if remove_n_points != 0 else each_run_y_array)

    return result_x_array, result_y_array


def find_time_for_target_accuracy(elements, target_accuracy_index, target_algo='express_flow'):
    target_accuracy = None
    results = {}

    # Find target accuracy for the algorithm
    for e in elements:
        x, y, algo = e
        # x = [val[1:] for val in x]  # Skip the first element in each list of x
        # y = [val[1:] for val in y]  # Skip the first element in each list of x
        x = [np.median(val) for val in zip(*x)]  # Calculate median per column
        y = [np.median(val) for val in zip(*y)]  # Calculate median per column
        if algo == target_algo:
            if target_accuracy_index < len(y):
                target_accuracy = y[target_accuracy_index]
            else:
                print(f"{target_algo} does not have a {target_accuracy_index}-th accuracy.")
                return
            break

    # Interpolate for other algorithms
    for e in elements:
        x, y, algo = e
        # x = [val[1:] for val in x]  # Skip the first element in each list of x
        # y = [val[1:] for val in y]  # Skip the first element in each list of x
        x = [np.median(val) for val in zip(*x)]  # Calculate median per column
        y = [np.median(val) for val in zip(*y)]  # Calculate median per column
        if target_accuracy > max(y):
            results[algo.split(" - ")[0]] = '-'
        else:
            estimated_time = np.interp(target_accuracy, y, x)
            results[algo.split(" - ")[0]] = estimated_time * 60

    return target_accuracy, results


def generate_and_draw_data(dataset):
    params = get_dataset_parameters(dataset)

    result_dir = "./internal/ml/model_selection/exp_result/"

    all_lines = []

    json_keys = [k for k, v in params.items() if isinstance(v, str) and v.endswith('.json')]

    for key in json_keys:
        result = read_json(params[key])

        sampled_x, sampled_y = sample_some_points(
            x_array=[result["sys_time_budget"] for _ in result["sys_acc"]],
            y_2d_array=result["sys_acc"],
            save_points=7,
            remove_n_points=0)

        all_lines.append([sampled_x, sampled_y, key])

    pprint(find_time_for_target_accuracy(elements=all_lines, target_accuracy_index=3))

    draw_structure_data_anytime(
        all_lines=all_lines,
        dataset=params['datasetfg_name'],
        name_img=f"{result_dir}/anytime_{dataset}",
        max_value=params['mx_value'],
        figure_size=params['figure_size'],
        annotations=params['annotations'],
        y_ticks=params['y_lim'],
        x_ticks=[0.01, None]
    )


# Choose dataset to process
dataset = "frappe"
generate_and_draw_data(dataset)
