

import json
import numpy as np
from matplotlib import pyplot as plt

import local_api
from benchmark_sampling import singleList
from draw import gather_all_run_result


def plot_experiment_compare(exp_m, label, title, axsp, plim=[]):

    for i in range(len(exp_m)):
        exp_m[i] = exp_m[i][:365]

    # 2. add simulate result
    exp = np.array(exp_m)
    q_75 = np.quantile(exp, .75, axis=0)
    q_25 = np.quantile(exp, .25, axis=0)
    mean = np.quantile(exp, .5, axis=0)
    print(label, mean[50], mean[100], mean[150], mean[200], mean[250], mean[300], mean[364])
    # plot simulate result
    axsp.plot(mean, label=label)
    axsp.fill_between(range(len(q_25)), q_25, q_75, alpha=0.1)

    # add title and grid
    axsp.set_title(title, fontsize=13)
    axsp.grid()
    axsp.legend()
    if len(plim) > 0:
        axsp.set_ylim([plim[0], plim[1]])


def compare_single(singleList, input_file_list, loapi, plim):

    f, allaxs = plt.subplots(2, 2, figsize=(15, 9))
    allaxs = allaxs.ravel()

    sampler_alg = {}
    for metrics in singleList:
        sampler_alg[metrics] = {}

    for input_file in input_file_list:
        with open(input_file, 'r') as readfile:
            all_run_result = json.load(readfile)
        sampler_name = input_file.split("/")[-1].split("_")[0]
        simulate_system_performance, acc_all_run = gather_all_run_result(all_run_result, 10, singleList, loapi)

        for metrics in singleList:
            if metrics in simulate_system_performance:
                sampler_alg[metrics][sampler_name] = simulate_system_performance[metrics]

    index = 0
    for metric, sampler_info in sampler_alg.items():
        for sampler_name, run_res in sampler_info.items():
            plot_experiment_compare(run_res, sampler_name, metric, allaxs[index], plim=plim)
        index += 1

    f.delaxes(allaxs[1])
    f.delaxes(allaxs[2])
    f.delaxes(allaxs[3])
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout()
    plt.show()
    f.savefig("compare.jpg", bbox_inches='tight')


def plot_experiment_compare_union(exp_m, label, title):
    for i in range(len(exp_m)):
        exp_m[i] = exp_m[i][:365]
    # 2. add simulate result
    exp = np.array(exp_m)
    q_75 = np.quantile(exp, .75, axis=0)
    q_25 = np.quantile(exp, .25, axis=0)
    mean = np.quantile(exp, .5, axis=0)
    # print(label, mean[50], mean[100], mean[150], mean[200], mean[250], mean[300], mean[364])
    # plot simulate result
    if title == "nas_wot_synflow":
        title = "vote"
        plt.plot(mean, label=title + "_" + label, linestyle='--')
    elif title == "synflow":
        plt.plot(mean, label= title+ "_"+label, linestyle=':')
    else:
        plt.plot(mean, label=title + "_" + label)
    plt.fill_between(range(len(q_25)), q_25, q_75, alpha=0.1)

    # add title and grid
    plt.grid()
    plt.legend()


def compare_union_one(singleList, input_file_list, loapi, plim):

    sampler_alg = {}
    for metrics in singleList:
        sampler_alg[metrics] = {}

    for input_file in input_file_list:
        with open(input_file, 'r') as readfile:
            all_run_result = json.load(readfile)
        sampler_name = input_file.split("/")[-1].split("_")[0]
        simulate_system_performance, acc_all_run = gather_all_run_result(all_run_result, 10, singleList, loapi)

        for metrics in singleList:
            if metrics in simulate_system_performance:
                sampler_alg[metrics][sampler_name] = simulate_system_performance[metrics]

    index = 0
    for metric, sampler_info in sampler_alg.items():
        for sampler_name, run_res in sampler_info.items():
            plot_experiment_compare_union(run_res, sampler_name, metric)
        index += 1

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout()
    plt.ylim(plim[0], plim[1])
    plt.title("Union Compare", fontsize=13)

    plt.savefig("compare.jpg", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    # input_file_list = ['./main/3_benchmark_sampler/201_cifar10_sampling_res/bohb_single',
    #                    './main/3_benchmark_sampler/201_cifar10_sampling_res/ea_single',
    #                    './main/3_benchmark_sampler/201_cifar10_sampling_res/random_single',
    #                    './main/3_benchmark_sampler/201_cifar10_sampling_res/rl_single',
    #                    ]

    # input_file_list = ['./main/3_benchmark_sampler/201_cifar10_sampling_res/bohb_vote',
    #                    './main/3_benchmark_sampler/201_cifar10_sampling_res/ea_vote',
    #                    './main/3_benchmark_sampler/201_cifar10_sampling_res/random_vote',
    #                    './main/3_benchmark_sampler/201_cifar10_sampling_res/rl_vote',
    #                    ]

    # input_file_list = ['./main/3_benchmark_sampler/201_imgNet_sampling_res/bohb_vote_ImageNet16-120',
    #                    './main/3_benchmark_sampler/201_imgNet_sampling_res/ea_vote_ImageNet16-120',
    #                    './main/3_benchmark_sampler/201_imgNet_sampling_res/random_vote_ImageNet16-120',
    #                    './main/3_benchmark_sampler/201_imgNet_sampling_res/rl_vote_ImageNet16-120',
    #                    ]

    # input_file_list = ['./main/3_benchmark_sampler/201_imgNet_sampling_res/bohb_single_ImageNet16-120',
    #                    './main/3_benchmark_sampler/201_imgNet_sampling_res/ea_single_ImageNet16-120',
    #                    './main/3_benchmark_sampler/201_imgNet_sampling_res/random_single_ImageNet16-120',
    #                    './main/3_benchmark_sampler/201_imgNet_sampling_res/rl_single_ImageNet16-120',
    #                    ]

    # input_file_list = ['./main/3_benchmark_sampler/101_cifar10_sampling_res/random_single_cifar10',
    #                    './main/3_benchmark_sampler/101_cifar10_sampling_res/rl_single_cifar10',
    #                    './main/3_benchmark_sampler/101_cifar10_sampling_res/ea_single_cifar10',
    #                    ]

    union_all_101_c10 = [
        './main/3_benchmark_sampler/101_cifar10_sampling_res/random_vote_cifar10',
        './main/3_benchmark_sampler/101_cifar10_sampling_res/rl_vote_cifar10',
        './main/3_benchmark_sampler/101_cifar10_sampling_res/ea_vote_cifar10',
        './main/3_benchmark_sampler/101_cifar10_sampling_res/random_single_cifar10',
        './main/3_benchmark_sampler/101_cifar10_sampling_res/rl_single_cifar10',
        './main/3_benchmark_sampler/101_cifar10_sampling_res/ea_single_cifar10',
        ]

    union_all_201_c10 = [
        './main/3_benchmark_sampler/201_cifar10_sampling_res/bohb_single',
        './main/3_benchmark_sampler/201_cifar10_sampling_res/ea_single',
        './main/3_benchmark_sampler/201_cifar10_sampling_res/random_single',
        './main/3_benchmark_sampler/201_cifar10_sampling_res/rl_single',
        './main/3_benchmark_sampler/201_cifar10_sampling_res/bohb_vote',
        './main/3_benchmark_sampler/201_cifar10_sampling_res/ea_vote',
        './main/3_benchmark_sampler/201_cifar10_sampling_res/random_vote',
        './main/3_benchmark_sampler/201_cifar10_sampling_res/rl_vote',
    ]

    union_all_201_imgNet = [
        './main/3_benchmark_sampler/201_imgNet_sampling_res/bohb_vote_ImageNet16-120',
        './main/3_benchmark_sampler/201_imgNet_sampling_res/ea_vote_ImageNet16-120',
        './main/3_benchmark_sampler/201_imgNet_sampling_res/random_vote_ImageNet16-120',
        './main/3_benchmark_sampler/201_imgNet_sampling_res/rl_vote_ImageNet16-120',
        './main/3_benchmark_sampler/201_imgNet_sampling_res/bohb_single_ImageNet16-120',
        './main/3_benchmark_sampler/201_imgNet_sampling_res/ea_single_ImageNet16-120',
        './main/3_benchmark_sampler/201_imgNet_sampling_res/random_single_ImageNet16-120',
        './main/3_benchmark_sampler/201_imgNet_sampling_res/rl_single_ImageNet16-120',
    ]

    space_name = "201"
    # space_name = "101"
    dataset = "c10"
    # dataset = "imgNet"

    if space_name == "201":
        if dataset == "c10":
            plim = [0.920, 0.945]
            pre_scored_data = "./result_sensitive/201_CIFAR10_15625/union/201_15625_c10_bs32_ic16_unionBest.json"
            file_usd = union_all_201_c10
        elif dataset == "imgNet":
            plim = [0.410, 0.470]
            pre_scored_data = "./result_bs32_ic16/IMAGENET/union/201_15k_imgNet_bs32_ic16_unionBest.json"
            file_usd = union_all_201_imgNet
        gt_file = "./201_200_result"
    elif space_name == "101":
        plim = [0.920, 0.945]
        pre_scored_data = "./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json"
        gt_file = "./101_result"
        file_usd = union_all_101_c10
    else:
        exit(0)

    loapi = local_api.LocalApi(pre_scored_data, gt_file, None, space_name)

    # singleList = ["grad_norm", "nas_wot", "synflow", "snip"]
    singleList = ["nas_wot", "synflow"]
    VoteList = ["nas_wot_synflow"]

    AllList = ["nas_wot_synflow", "nas_wot", "synflow"]

    # compare_single(VoteList, input_file_list, loapi, plim)
    compare_union_one(AllList, file_usd, loapi, plim)


