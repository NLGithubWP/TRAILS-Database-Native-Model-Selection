import numpy as np

from utilslibs.io_tools import write_json, read_json
from matplotlib import pyplot as plt
import matplotlib


# lines' mark size
set_marker_size = 10
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 30
set_lgend_size = 15
set_tick_size = 15

frontinsidebox = 23

# update tick size
matplotlib.rc('xtick', labelsize=set_tick_size)
matplotlib.rc('ytick', labelsize=set_tick_size)

plt.rcParams['axes.labelsize'] = set_tick_size

mark_list = ["o", "*", "<", "^", "s", "d", "D", ">", "h"]

line_shape_list = ['-.', '--', '-', ':']


def add_one_line(x_time_array, y_twod_budget, namespace):
    # training-based
    x_ = x_time_array
    y_ = y_twod_budget

    exp = np.array(y_) * 100
    y_h = np.quantile(exp, .75, axis=0)
    y_m = np.quantile(exp, .5, axis=0)
    y_l = np.quantile(exp, .25, axis=0)

    plt.fill_between(x_, y_l, y_h, alpha=0.1)
    plt.plot(x_, y_m, "o-", label=namespace)

# 'frappe, criteo, uci_diabetes'
dataset = "frappe"

train_result19 = read_json(f"./res_train_base_line_{dataset}_epoch_19.json")
train_result5 = read_json(f"./res_train_base_line_{dataset}_epoch_5.json")
train_result1 = read_json(f"./res_train_base_line_{dataset}_epoch_1.json")
system_result = read_json(f"./res_end_2_end_{dataset}_10e4.json")
training_free_result = read_json(f"./res_end_2_end_{dataset}_p1.json")

# training-based
add_one_line(train_result19["baseline_time_budget"][0], train_result19["baseline_acc"], "Train-Based-FullyTrain")
add_one_line(train_result5["baseline_time_budget"][0], train_result5["baseline_acc"], "Train-Based-EarlyStopping")
# add_one_line(train_result1["baseline_time_budget"][0], train_result1["baseline_acc"], "Train-Based-1e")
add_one_line(system_result["sys_time_budget"], system_result["sys_acc"], "FIRMEST")
add_one_line(training_free_result["sys_time_budget"], training_free_result["sys_acc"], "Train-Free")


plt.xlim(-1, 1e5/2)
# plt.ylim(97.5, 98.1)

plt.tight_layout()
plt.xscale("symlog")
plt.grid()
plt.xlabel("Time Budget given by user (min)", fontsize=set_font_size)
plt.ylabel(f"Test accuracy on {dataset}", fontsize=set_font_size)
plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=set_lgend_size)
# plt.show()
plt.savefig(f"./any_time_{dataset}.pdf", bbox_inches='tight')






