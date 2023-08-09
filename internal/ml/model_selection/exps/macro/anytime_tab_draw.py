from typing import List

from src.tools.compute import sample_in_log_scale_new
from src.tools.io_tools import read_json
from exps.draw_tab_lib import draw_structure_data_anytime

trian_time = 3
def get_dataset_parameters(dataset):
    parameters = {
        "uci_diabetes": {
            "epoch": 0,
            "sys_end2end_res": "./internal/ml/model_selection/exp_result/res_end_2_end_uci_diabetes_100_5.json",
            "sys_end2end_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_uci_diabetes_100_5_p1.json",
            "tab_nas_res": "./tabNAS_benchmark_uci_diabetes_epoch_0.json",
            "mx_value": 67.4,
            "y_lim": [61.8, 67.5],
            "figure_size": (6.2, 4.71),
            "datasetfg_name": "Diabetes",
            "annotations": [],  # ["TabNAS", 63.33, 8.14/60],
            "remove_n_points": 1,
        },
        "frappe": {
            "epoch": 19,
            "sys_end2end_res": "./internal/ml/model_selection/exp_result/res_end_2_end_frappe_100_5.json",
            "sys_end2end_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_frappe_100_5_p1.json",
            "tab_nas_res": "./tabNAS_benchmark_frappe_epoch_19.json",
            "mx_value": 98.052,
            "y_lim": [97.6, None],
            "figure_size": (6.2, 4.71),
            "datasetfg_name": dataset,
            "annotations": [],  # ["TabNAS", 97.68, 324.8/60],
            "remove_n_points": 2,
        },
        "criteo": {
            "epoch": 9,
            "sys_end2end_res": "./internal/ml/model_selection/exp_result/res_end_2_end_criteo_100_5.json",
            "sys_end2end_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_criteo_100_5_p1.json",
            "tab_nas_res": "./tabNAS_benchmark_criteo_epoch_9.json",
            "mx_value": 80.335,
            "y_lim": [80.1, None],
            "figure_size": (6.2, 4.71),
            "datasetfg_name": dataset,
            "annotations": [],  # ["TabNAS", 80.17, 7250.0/60],
            "remove_n_points": 3,
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


def generate_and_draw_data(dataset):
    params = get_dataset_parameters(dataset)
    if not params:
        print(f"No parameters for the dataset: {dataset}")
        return

    result_dir = "./internal/ml/model_selection/exp_result/"

    train_based_res = read_json(
        f"{result_dir}/res_train_base_line_{dataset}_epoch_{params['epoch']}.json")
    sampled_train_x, sampled_train_y = sample_some_points(x_array=train_based_res["baseline_time_budget"],
                                                          y_2d_array=train_based_res["baseline_acc"],
                                                          save_points=9,
                                                          remove_n_points=params['remove_n_points'])

    system_result = read_json(params['sys_end2end_res'])
    system_p1_result = read_json(params['sys_end2end_p1'])
    tab_nas_res = read_json(params["tab_nas_res"])

    sampled_sys_x, sampled_sys_y = sample_some_points(
        x_array=[system_result["sys_time_budget"] for _ in system_result["sys_acc"]],
        y_2d_array=system_result["sys_acc"],
        save_points=7,
        remove_n_points=0)

    tabnas_x, tabnas_y = sample_some_points(
        x_array=[[earch * trian_time for earch in ele] for ele in tab_nas_res["baseline_time_budget"]],
        y_2d_array=tab_nas_res["baseline_acc"],
        save_points=9,
        remove_n_points=3)

    all_lines = [
        [sampled_train_x, sampled_train_y, "Training-Based MS"],
        [system_p1_result["sys_time_budget"], system_p1_result["sys_acc"], "Training-Free MS"],
        [sampled_sys_x, sampled_sys_y, "2Phase-MS"],
        [tabnas_x, tabnas_y, "TabNAS"],
    ]

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
# dataset = "uci_diabetes"
# dataset = "criteo"
generate_and_draw_data(dataset)

