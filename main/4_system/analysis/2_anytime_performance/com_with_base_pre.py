
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from common.constant import Config
from query_api.db_ops import fetch_from_db
from query_api.parse_pre_res import FetchGroundTruth
from utilslibs.tools import write_json, read_json


def plt_running_cfg():

    # fastAutoNAS
    y_sys_list = []
    x_sys_list = []

    # training-based
    x_list_train_10 = []

    # fastAutoNAS GT
    y_sys_gt_list = []

    for run in range(2):
        x_axis_each_run = []
        y_axis_each_run = []
        x_train_10_each_run = []
        # for sys gt
        sys_gt_list = []
        current_best = 0
        # begin to run !
        begin_run = time.time()
        for model_id in range(1, 15626):

            arch_id, candidates, curr_time = fetch_from_db(run, model_id)
            x_axis_each_run.append(curr_time)

            # training each one
            # query_begin = time.time()
            acc, train_time_usage = fgt.get_high_acc_top_10(candidates)
            # print(time.time()-query_begin)

            y_axis_each_run.append(acc)
            x_train_10_each_run.append(train_time_usage)

            score_, _ = fgt.get_ground_truth(arch_id)
            if score_ > current_best:
                current_best = score_
            sys_gt_list.append(current_best)
        print(run, "time usage =", time.time() - begin_run)
        x_sys_list.append(x_axis_each_run)
        x_list_train_10.append(x_train_10_each_run)
        y_sys_list.append(y_axis_each_run)
        y_sys_gt_list.append(sys_gt_list)

    mean_x = np.quantile(np.array(x_sys_list), .5, axis=0)
    mean_x_with_train = np.quantile(np.array(x_list_train_10), .5, axis=0) + mean_x
    # this is to compare efficiency
    exp = np.array(y_sys_list)
    q_75_acc = np.quantile(exp, .75, axis=0)
    q_25_cc = np.quantile(exp, .25, axis=0)
    mean_acc = np.quantile(exp, .5, axis=0)
    plt.fill_between(mean_x_with_train, q_25_cc, q_75_acc, alpha=0.3)
    plt.plot(mean_x_with_train, mean_acc, label="TFMEM-phase2")
    plt.plot(mean_x, mean_acc, "--", label="TFMEM-phase1")

    exp = np.array(y_sys_gt_list)
    q_75_acc = np.quantile(exp, .75, axis=0)
    q_25_acc = np.quantile(exp, .25, axis=0)
    mean_acc = np.quantile(exp, .5, axis=0)
    plt.fill_between(mean_x_with_train.tolist(), q_25_acc, q_75_acc, alpha=0.3)
    plt.plot(mean_x_with_train.tolist(), mean_acc, label="Accuracy")

    plot_res = {"mean_x": mean_x.tolist(),
                "y_sys_list": y_sys_list,
                "mean_x_with_train": mean_x_with_train.tolist(),
                "y_sys_gt_list": y_sys_gt_list}

    write_json("plt_score_", plot_res)


def plt_training_based_cfg():
    begin_time = time.time()
    x_list = []
    y_list = []
    # each run evaluates 300 models
    for run, info in train_based_result.items():
        # average x axis
        x_list.append(info["x_axis_time"])
        y_list.append([ele * 0.01 for ele in info["current_best_acc"]])

    mean_x = np.quantile(np.array(x_list), .5, axis=0)

    # this is to trainint-based efficiency
    exp = np.array(y_list)
    q_75 = np.quantile(exp, .75, axis=0)
    q_25 = np.quantile(exp, .25, axis=0)
    mean = np.quantile(exp, .5, axis=0)
    plt.fill_between(mean_x, q_25, q_75, alpha=0.3)
    plt.plot(mean_x, mean, label="Training-based Model Selection")

    plot_res = {"x_": mean_x.tolist(), "y_": y_list}
    write_json("plt_trained_based_", plot_res)
    print("plt_training_based_cfg: ", time.time() - begin_time)


def plt_with_pre_saved_data():
    score_data = read_json("plt_score_")

    x_list = score_data["mean_x"]
    y_list = score_data["y_sys_list"]
    mean_x_with_train = score_data["mean_x_with_train"]

    exp = np.array(y_list)
    q_75_acc = np.quantile(exp, .75, axis=0)
    q_25_cc = np.quantile(exp, .25, axis=0)
    mean_acc = np.quantile(exp, .5, axis=0)
    plt.fill_between(x_list, q_25_cc, q_75_acc, alpha=0.3)
    plt.plot(mean_x_with_train, mean_acc, label="TFMEM-phase2")
    plt.plot(x_list, mean_acc, label="TFMEM-phase1")

    y_list = score_data["y_sys_gt_list"]
    exp = np.array(y_list)
    q_75 = np.quantile(exp, .75, axis=0)
    q_25 = np.quantile(exp, .25, axis=0)
    mean = np.quantile(exp, .5, axis=0)
    plt.fill_between(x_list, q_25, q_75, alpha=0.3)
    plt.plot(x_list, mean,  label="TFMEM-GT")

    # this is training based
    train_data = read_json("plt_trained_based_")
    x_list = [ele for ele in train_data["x_"] if ele < 86400]
    y_list = [ele[:len(x_list)] for ele in train_data["y_"]]
    exp = np.array(y_list)
    q_75 = np.quantile(exp, .75, axis=0)
    q_25 = np.quantile(exp, .25, axis=0)
    mean = np.quantile(exp, .5, axis=0)
    plt.fill_between(x_list, q_25, q_75, alpha=0.3)
    plt.plot(x_list, mean, "X-", label="Training-based MS")


if __name__ == "__main__":
    base_dir = os.getcwd()
    frontsizeall = 15

    save_img_path = os.path.join(base_dir, "time_budget_com.pdf")
    space = Config.NB201
    fgt = FetchGroundTruth(space)

    train_based_result = read_json(os.path.join(base_dir,
                                                "result_base/result_system/simulate/train_based_201_200run_3km_ea.json"))

    # plt_running_cfg()
    # plt_training_based_cfg()

    plt_with_pre_saved_data()
    #
    plt.xscale("symlog")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.title("Model Selection (MS)")
    plt.xlabel("Time Budget T (Second)", fontsize=frontsizeall)
    plt.ylabel("Highest Test Accuracy (%)", fontsize=frontsizeall)
    # plt.ylim()
    plt.show()
    # plt.savefig(save_img_path, bbox_inches='tight')
