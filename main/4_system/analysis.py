
import json
import numpy as np
import matplotlib.pyplot as plt
from search_space.nas_101_api.lib import nb101_api
from search_space import NasBench201Space, NasBench101Cfg, NasBench101Space
from search_space.nas_201_api.lib import nasbench2, NASBench201API


def get_ground_truth(arch_id):
    score_, _ = api.query_performance(int(arch_id), "cifar10")
    # score_, _ = NasBench201Space.simulate_tran_eval_log(api, int(arch_id), "cifar10", iepoch=None, hp="12")
    return score_


def get_running_cfg():
    x_list = []
    y_list = []

    # each run evaluates 300 models
    for i in range(total_models):
        x_list.append(0)
    for run, info in result.items():
        # average x axis
        for i in range(len(info["x_axis_time"])):
            x_list[i] += info["x_axis_time"][i]

        # get accuracy of y axis
        y_axis_each_run = []
        for top10 in info["y_axis_top10_models"]:
            y_axis_each_run.append(get_high_acc_top_10(top10))
        y_list.append(y_axis_each_run)

    x_list = [ele/total_models for ele in x_list]
    return x_list, y_list


def get_high_acc_top_10(top10):
    current_best = 0
    for arch_id in top10:
        score_ = get_ground_truth(arch_id)
        if score_ > current_best:
            current_best = score_
    return current_best


def gt_cfg():
    acc = []
    for run, info in result.items():
        run_score_list = []
        curr_run_best = 0
        for i in range(len(info["arch_id_list"])):
            arch_id = info["arch_id_list"][i]
            score_ = get_ground_truth(arch_id)
            if score_ > curr_run_best:
                curr_run_best = score_
            run_score_list.append(curr_run_best)
        acc.append(run_score_list)
    return acc


with open("res_scoring.json", 'r') as outfile:
    result = json.load(outfile)

total_models = 20

# 201
# api_loc = "/Users/kevin/project_python/Fast-AutoNAS/data/NAS-Bench-201-v1_0-e61699.pth"
# api = NASBench201API(api_loc)

# 101
api_loc = "/Users/kevin/project_python/Fast-AutoNAS/data/nasbench_only108.pkl"
model_cfg = NasBench101Cfg(16, 3, 3, 10, True)
api = NasBench101Space(api_loc, model_cfg)


x_list, y_list = get_running_cfg()


acc = gt_cfg()
exp = np.array(acc)
q_75 = np.quantile(exp, .75, axis=0)
q_25 = np.quantile(exp, .25, axis=0)
mean = np.quantile(exp, .5, axis=0)
plt.fill_between(x_list, q_25, q_75, alpha=0.3)
plt.plot(x_list, mean)


exp = np.array(y_list)
q_75 = np.quantile(exp, .75, axis=0)
q_25 = np.quantile(exp, .25, axis=0)
mean = np.quantile(exp, .5, axis=0)
plt.fill_between(x_list, q_25, q_75, alpha=0.3)
plt.plot(x_list, mean)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
