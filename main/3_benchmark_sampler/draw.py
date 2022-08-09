

import json
from matplotlib import pyplot as plt
import numpy as np
import local_api
from benchmark_sampling import singleList, voteList


def plot_acc(accuracy, axsp, plim=[]):

    for i in range(len(accuracy)):
        accuracy[i] = accuracy[i][:365]

    # 1. add accuracy
    accuracy_exp = np.array(accuracy)
    accuracy_q_75 = np.quantile(accuracy_exp, .75, axis=0)
    accuracy_q_25 = np.quantile(accuracy_exp, .25, axis=0)
    accuracy_mean = np.quantile(accuracy_exp, .5, axis=0)
    # plot accuracy
    axsp.plot(accuracy_mean, label="test_acc")
    axsp.fill_between(range(len(accuracy_q_25)), accuracy_q_25, accuracy_q_75, alpha=0.1)

    # add title and grid
    axsp.grid()
    axsp.legend()
    if len(plim) > 0:
        axsp.set_ylim([plim[0], plim[1]])


def plot_experiment(exp_m, label, title, axsp, plim=[]):

    for i in range(len(exp_m)):
        exp_m[i] = exp_m[i][:365]

    # 2. add simulate result
    exp = np.array(exp_m)

    try:
        q_75 = np.quantile(exp, .75, axis=0)
        q_25 = np.quantile(exp, .25, axis=0)
        mean = np.quantile(exp, .5, axis=0)
    except:
        print(exp)

    # plot simulate result
    axsp.plot(mean, label=label)
    axsp.fill_between(range(len(q_25)), q_25, q_75, alpha=0.1)

    # add title and grid
    axsp.set_title(title, fontsize=13)
    axsp.grid()
    axsp.legend()
    if len(plim) > 0:
        axsp.set_ylim([plim[0], plim[1]])


def gather_all_run_result(all_run_result, top_k, singleList, loapi):
    simulate_system_performance = {}
    best_acc_all_run = {}
    for alg_name in singleList:
        if alg_name not in all_run_result["0"]:
            continue
        simulate_system_performance[alg_name] = []
        best_acc_all_run[alg_name] = []
        for _, run_info in all_run_result.items():
            simulate_system_result = loapi.api_simulate_evaluate(
                run_info[alg_name]["acc"], run_info[alg_name]["ori_score"], top_k)
            best_acc = loapi.api_get_acc(run_info[alg_name]["acc"])
            simulate_system_performance[alg_name].append(simulate_system_result[:380])
            best_acc_all_run[alg_name].append(best_acc[:380])
    return simulate_system_performance, best_acc_all_run


def gather_all_run_result_vote(all_run_result, top_k, loapi):
    simulate_system_performance = {}
    best_acc_all_run = {}
    for vote_com in voteList:
        vote_comb_name = "_".join(vote_com)
        simulate_system_performance[vote_comb_name] = []
        best_acc_all_run[vote_comb_name] = []
        for _, run_info in all_run_result.items():
            simulate_system_result = loapi.api_simulate_evaluate(
                run_info[vote_comb_name]["acc"], run_info[vote_comb_name]["ori_score"], top_k)
            best_acc = loapi.api_get_acc(run_info[vote_comb_name]["acc"])
            simulate_system_performance[vote_comb_name].append(simulate_system_result)
            best_acc_all_run[vote_comb_name].append(best_acc)
    return simulate_system_performance, best_acc_all_run


def draw_graph_vote(simulate_system_performance, acc_all_run, img_name, plim=[]):

    f, allaxs = plt.subplots(1, 2, figsize=(15, 9))
    allaxs = allaxs.ravel()
    index = 0

    voteListMapper = {"_".join(voteList[i]):  i for i in range(len(voteList))}

    for alg_name in simulate_system_performance:
        label = voteListMapper[alg_name]
        plot_experiment(simulate_system_performance[alg_name],
                        label,
                        label,
                        allaxs[index],
                        plim)
        plot_acc(acc_all_run[alg_name], allaxs[index], plim=plim)
        index += 1
    f.delaxes(allaxs[1])
    plt.grid()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout()
    plt.show()
    f.savefig(img_name, bbox_inches='tight')


if __name__ == '__main__':

    input_file = './main/3_benchmark_sampler/101_cifar10_sampling_res/bohb_single_cifar10'
    pre_scored_data = "./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json"
    gt_file = "./101_result"
    train_num_list = [1, 5, 10, 20]
    train_num_one = 10
    plim = [0.90, 0.94]
    img_name = input_file + ".jpg"

    with open(input_file, 'r') as readfile:
        all_run_result = json.load(readfile)

    loapi = local_api.LocalApi(pre_scored_data, gt_file, None, "101")

    if "vote" in input_file:
        simulate_system_performance, acc_all_run = gather_all_run_result_vote(all_run_result, train_num_one, loapi)
        draw_graph_vote(simulate_system_performance, acc_all_run, input_file+"{}.jpg".format(train_num_one), plim=plim)
    else:

        # the metrics to draw
        # singleList = ["grad_norm", "nas_wot", "grasp", "synflow", "snip", "ntk_trace", "fisher", "ntk_trace_approx"]
        # singleList = ["grad_norm", "nas_wot", "synflow", "snip"]
        singleList = ["nas_wot", "synflow"]
        # singleList = ["nas_wot"]

        sampler_alg = {}
        for metrics in singleList:
            sampler_alg[metrics] = {}

        f, allaxs = plt.subplots(1, 2, figsize=(15, 9))
        allaxs = allaxs.ravel()

        # collect all ac![](101_cifar10_sampling_res/ea_vote_cifar1010.jpg)c over different top K
        acc_all_run = []
        for train_num in train_num_list:
            simulate_system_performance, acc_all_run = \
                gather_all_run_result(all_run_result, train_num, singleList, loapi)

            for metrics in singleList:
                sampler_alg[metrics][train_num] = simulate_system_performance[metrics]

        # draw
        index = 0
        for metric, sampler_info in sampler_alg.items():
            for train_k, run_res in sampler_info.items():
                plot_experiment(run_res, "top_{}".format(train_k), metric, allaxs[index], plim=plim)

            plot_acc(acc_all_run[metric], allaxs[index], plim=plim)
            index += 1

        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.tight_layout()
        plt.show()
        f.savefig(img_name, bbox_inches='tight')

