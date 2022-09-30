
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from api_local.parse_pre_res import FetchGroundTruth
from common.constant import Config
from utilslibs.tools import write_json, read_json


def plt_running_cfg():
    begin_time = time.time()
    x_list = []
    y_list = []
    x_list_train_10 = []

    for run, info in result.items():
        # average x axis
        x_list.append(info["x_axis_time"])
        # get accuracy of y axis
        y_axis_each_run = []
        x_train_10_each_run = []
        for top10 in info["y_axis_top10_models"]:
            acc, time_usg = fgt.get_high_acc_top_10(top10)
            y_axis_each_run.append(acc)
            x_train_10_each_run.append(time_usg)
        x_list_train_10.append(x_train_10_each_run)
        # record each run's acc
        y_list.append(y_axis_each_run)

    mean_x = np.quantile(np.array(x_list), .5, axis=0)
    mean_x_with_train = np.quantile(np.array(x_list_train_10), .5, axis=0)+mean_x
    # this is to compare efficiency
    exp = np.array(y_list)
    q_75_acc = np.quantile(exp, .75, axis=0)
    q_25_cc = np.quantile(exp, .25, axis=0)
    mean_acc = np.quantile(exp, .5, axis=0)
    plt.fill_between(mean_x_with_train, q_25_cc, q_75_acc, alpha=0.3)
    plt.plot(mean_x_with_train, mean_acc, label="TFMEM-phase2")
    plt.plot(mean_x, mean_acc, "--", label="TFMEM-phase1")

    plot_res = {"x_": mean_x.tolist(), "y_": y_list, "mean_x_with_train": mean_x_with_train.tolist()}
    write_json("plt_score_", plot_res)

    print("plt_running_cfg: ", time.time() - begin_time)
    return mean_x_with_train.tolist()


def plt_gt_cfg(x_list):
    begin_time = time.time()
    acc = []
    for run, info in result.items():
        run_score_list = []
        current_best = 0
        for i in range(len(info["arch_id_list"])):
            arch_id = info["arch_id_list"][i]
            score_, _ = fgt.get_ground_truth(arch_id)
            if score_ > current_best:
                current_best = score_
            run_score_list.append(current_best)
        acc.append(run_score_list)

    # this is to compare effectiveness
    exp = np.array(acc)
    q_75_acc = np.quantile(exp, .75, axis=0)
    q_25_acc = np.quantile(exp, .25, axis=0)
    mean_acc = np.quantile(exp, .5, axis=0)
    plt.fill_between(x_list, q_25_acc, q_75_acc, alpha=0.3)
    plt.plot(x_list, mean_acc, label="Accuracy" )

    plot_res = {"x_": x_list, "y_": acc}
    write_json("plt_score_gt_", plot_res)

    print("plt_gt_cfg: ", time.time() - begin_time)
    return acc


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
    plt.plot(mean_x, mean,  label="Training-based Model Selection" )

    plot_res = {"x_": mean_x.tolist(), "y_": y_list}
    write_json("plt_trained_based_", plot_res)
    print("plt_training_based_cfg: ", time.time() - begin_time)


def plt_with_pre_saved_data():

    score_data = read_json("plt_score_")
    train_data = read_json("plt_trained_based_")
    # score_gt = read_json("plt_score_gt_")

    x_list = score_data["x_"]
    y_list = score_data["y_"]
    mean_x_with_train = score_data["mean_x_with_train"]

    exp = np.array(y_list)
    q_75_acc = np.quantile(exp, .75, axis=0)
    q_25_cc = np.quantile(exp, .25, axis=0)
    mean_acc = np.quantile(exp, .5, axis=0)
    plt.fill_between(x_list, q_25_cc, q_75_acc, alpha=0.3)
    # plt.plot(mean_x_with_train, mean_acc, label="TFMEM-based MS, phase2")
    plt.plot(x_list, mean_acc, label="TFMEM-based MS")

    # this is training based

    x_list = [ele for ele in train_data["x_"] if ele < 86400]
    y_list = [ele[:len(x_list)] for ele in train_data["y_"]]

    exp = np.array(y_list)
    q_75 = np.quantile(exp, .75, axis=0)
    q_25 = np.quantile(exp, .25, axis=0)
    mean = np.quantile(exp, .5, axis=0)
    plt.fill_between(x_list, q_25, q_75, alpha=0.3)
    plt.plot(x_list, mean, "X-", label="Training-based MS")
    # plt.text(86400, 0.87, s="One Day Budget")

    # x_list = score_gt["x_"]
    # y_list = score_gt["y_"]
    # exp = np.array(y_list)
    # q_75 = np.quantile(exp, .75, axis=0)
    # q_25 = np.quantile(exp, .25, axis=0)
    # mean = np.quantile(exp, .5, axis=0)
    # plt.fill_between(x_list, q_25, q_75, alpha=0.3)
    # plt.plot(x_list, mean,  label="TFMEM-GT" )


if __name__ == "__main__":
    frontsizeall = 15
    base_dir = os.getcwd()

    save_img_path = os.path.join(base_dir, "time_budget_com.pdf")
    # space = Config.NB201
    # fgt = FetchGroundTruth(space)
    # result = read_json(os.path.join(base_dir,
    #     "result_base/result_system/simulate/TFMEM_201_200run_3km_ea.json"))
    # total_models = len(result["0"]["arch_id_list"])
    # train_based_result = read_json(os.path.join(base_dir,
    #     "result_base/result_system/simulate/train_based_201_200run_3km_ea.json"))
    #
    # x_list = plt_running_cfg()
    # plt_gt_cfg(x_list)
    # plt_training_based_cfg()

    plt_with_pre_saved_data()

    # plt.hlines(0.93, xmin=1, xmax=1000000, linewidth=1, linestyle='--')

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
