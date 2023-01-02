
from matplotlib import pyplot as plt
from common.constant import Config
import matplotlib

from plot_libs.graph_lib import export_legend
from utilslibs.tools import read_json

frontsizeall = 20
marksizeall = 15
# set_marker_size = 12
# points' mark size
set_tick_size = 17
# update tick size
matplotlib.rc('xtick', labelsize=set_tick_size)
matplotlib.rc('ytick', labelsize=set_tick_size)

plt.rcParams['axes.labelsize'] = set_tick_size

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


mark_list = ["o-", "*-", "<-"]
key_list = ["uniform", "sr", "sh"]
index = 0
for key in key_list:
    value = result_save_dic[key]
    time_used_mean = value["time_used_mean"]
    accuracy_mean = value["accuracy_mean"]
    accuracy_q_25 = value["accuracy_q_25"]
    accuracy_q_75 = value["accuracy_q_75"]

    if key == "sh":
        The_name = "SUCCHALF"
        new_time_used_mean = [47.95, 89.53333333333333, 185.81666666666666, 409.28333333333336, 582.5333333333333, 798.7666666666667, 1015.0, 1445.0, 1661.6666666666667]
        new_index_array = [time_used_mean.index(ele) for ele in new_time_used_mean]
        new_accuracy_mean = [accuracy_mean[i] for i in new_index_array]
        new_accuracy_q_25 = [accuracy_q_25[i] for i in new_index_array]
        new_accuracy_q_75 = [accuracy_q_75[i] for i in new_index_array]
    elif key == "uniform":
        The_name = "UNIFORM"
        new_time_used_mean = [8.333333333333334, 58.333333333333336, 116.66666666666667, 333.3333333333333, 558.3333333333334, 775.0, 1108.3333333333333, 1500.0, 1658.3333333333333]
        new_index_array = [time_used_mean.index(ele) for ele in new_time_used_mean]
        new_accuracy_mean = [accuracy_mean[i] for i in new_index_array]
        new_accuracy_q_25 = [accuracy_q_25[i] for i in new_index_array]
        new_accuracy_q_75 = [accuracy_q_75[i] for i in new_index_array]
    else:
        The_name = "SUCCREJCT"
        new_time_used_mean = [1865.1333333333334, 5092.566666666667, 12449.35, 20035.166666666668, 23862.966666666667
                              ]
        new_index_array = [time_used_mean.index(ele) for ele in new_time_used_mean]
        new_accuracy_mean = [accuracy_mean[i] for i in new_index_array]
        new_accuracy_q_25 = [accuracy_q_25[i] for i in new_index_array]
        new_accuracy_q_75 = [accuracy_q_75[i] for i in new_index_array]

    plt.plot(new_time_used_mean, new_accuracy_mean, mark_list[index], label=The_name, markersize=marksizeall)
    plt.fill_between(new_time_used_mean, new_accuracy_q_25, new_accuracy_q_75, alpha=0.1)
    index += 1


plt.ylim(sub_graph_y1[0], sub_graph_y1[1])
plt.xscale("symlog")
plt.tight_layout()
plt.grid()
plt.xlabel(r"Time Budget $T$ (min)", fontsize=frontsizeall)
plt.ylabel(f"Test Accuracy on {img_in_graph}", fontsize=frontsizeall)
# plt.legend(ncol=3, prop={'size': 14.5})
# plt.legend(ncol=3)
# plt.show()
export_legend(fig2, filename="phase2_micro_legend", unique_labels=["UNIFORM", "SUCCREJCT", "SUCCHALF"])
fig2.savefig(f"phase2_micro_{dataset}.pdf", bbox_inches='tight')
