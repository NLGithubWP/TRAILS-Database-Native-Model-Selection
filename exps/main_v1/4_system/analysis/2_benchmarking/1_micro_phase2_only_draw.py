from typing import List

from matplotlib import pyplot as plt
from common.constant import Config
import matplotlib

from utilslibs.compute import sample_in_log_scale_new
from utilslibs.draw_lib import draw_structure_data_anytime
from utilslibs.draw_tools import export_legend
from utilslibs.io_tools import read_json


search_space = "nasbench201"
dataset = "cifar10"
# dataset = "ImageNet16-120"

if dataset == Config.imgNet:
    img_in_graph = "ImageNet"
elif dataset == Config.c10:
    img_in_graph = "CIFAR10"
else:
    exit(1)


# min, t
if dataset == Config.c10:
    # C10 array
    sub_graph_y1 = [93.4, 94.2]
elif dataset == Config.c100:
    # C10 array
    sub_graph_y1 = [64, 74]
else:
    # ImgNet X array
    sub_graph_y1 = [44, 46.5]


result_save_dic = read_json(f"micro_phase2_{dataset}")
fig2 = plt.figure(figsize=(7, 5))


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


mark_list = ["o-", "*-", "<-"]
key_list = ["uniform", "sr", "sh"]

all_lines = []
for key in key_list:
    value = result_save_dic[key]
    time_used = value["time_used"]
    acc_reached = value["acc_reached"]
    if key == "sh":
        The_name = "SUCCHALF"
        sampled_train_x, sampled_train_y = sample_some_points(
           x_array=time_used,
           y_2d_array=acc_reached,
           save_points=5,
           remove_n_points=0)
    elif key == "uniform":
        The_name = "UNIFORM"
        sampled_train_x, sampled_train_y = sample_some_points(
           x_array=time_used,
           y_2d_array=acc_reached,
           save_points=5,
           remove_n_points=0)
    else:
        The_name = "SUCCREJCT"
        sampled_train_x, sampled_train_y = sample_some_points(
           x_array=time_used,
           y_2d_array=acc_reached,
           save_points=5,
           remove_n_points=0)
    inner_res = [sampled_train_x, sampled_train_y, The_name]
    all_lines.append(inner_res)


draw_structure_data_anytime(
    all_lines=all_lines,
    dataset="C10",
    name_img=f"phase2_micro_{dataset}",
    max_value=-1,
    y_ticks=sub_graph_y1,
    x_ticks=None
)

# export_legend(fig2, filename="phase2_micro_legend", unique_labels=["UNIFORM", "SUCCREJCT", "SUCCHALF"])
# fig2.savefig(f"phase2_micro_{dataset}.pdf", bbox_inches='tight')
