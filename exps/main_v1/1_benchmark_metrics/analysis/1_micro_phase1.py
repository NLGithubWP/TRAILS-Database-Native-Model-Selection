import copy
import json
import os
import scipy.stats as ss
import numpy as np
from os.path import exists
from matplotlib import pyplot as plt

from exps.main_v1.statistic_lib import sort_update_with_batch_average, get_rank_after_sort, \
    sort_update_with_batch_average_hlm, sort_update
from utilslibs.measure_tools import CorCoefficient
from query_api.query_model_gt_acc_api import Gt101


def union_best_bn_cfg(bn_input_file_path, noBn_input_file_path, output_file_path):
    if exists(output_file_path):
        return
    # read bn and no-bn file
    with open(bn_input_file_path, 'r') as readfile:
        data_bn = json.load(readfile)
    with open(noBn_input_file_path, 'r') as readfile:
        data_no_bn = json.load(readfile)

    # replace bn with no-bn
    new_data = {}
    all_keys = set(data_no_bn.keys()).intersection(set(data_bn.keys()))
    print(len(all_keys), len(data_no_bn.keys()), len(data_bn.keys()))
    for arch_id in list(all_keys):
        bn_info = data_bn[arch_id]
        no_bn_info = data_no_bn[arch_id]

        new_data[arch_id] = bn_info
        new_data[arch_id]["synflow"] = no_bn_info["synflow"]

    # write to file
    with open(output_file_path, 'w') as outfile:
        outfile.write(json.dumps(new_data))


def add_vote_info(input_file_path, output_file):
    if exists(output_file):
        return
    # update single metrics's score into rank
    def update_score_to_rank(data, algName):
        archIds = []
        res = []
        for archId, info in data.items():
            if algName in info:
                res.append(float(info[algName]))
                archIds.append(archId)
        ranked_res = ss.rankdata(res)

        # update origin dict
        for archId, info in data.items():
            if algName in info:
                info[algName] = str(int(ranked_res[archIds.index(archId)]))

    # add score => vote score
    def add_metrics_score(info, combs):
        res = 0
        for algName in combs:
            res += int(info[algName])
        return str(res)

    all_ss = ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"]

    with open(input_file_path, 'r') as readfile:
        data = json.load(readfile)

    for alg_name in all_ss:
        update_score_to_rank(data, alg_name)

    all_keys = data.keys()
    print(len(all_keys))

    total_vote_combination = [["nas_wot", "synflow"]]
    for archId, info in data.items():
        for comb in total_vote_combination:
            info[str(comb)] = {}
            info[str(comb)] = add_metrics_score(info, comb)

    with open(output_file, 'w') as outfile:
        outfile.write(json.dumps(data))


def partition_key_int_groups(input_file, output_file):
    with open(input_file, 'r') as readfile:
        data = json.load(readfile)

    visited = {}
    num_dist = 0

    num_log = 1000
    num_log_index = 0

    all_archs = list(data.keys())
    all_ss = [
        "grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"]

    arch_with_all_metrics = []
    for arch in all_archs:
        isContinue = 1
        for ele in all_ss:
            if ele not in data[arch]:
                isContinue = 0
                break
        if isContinue == 0:
            continue
        arch_with_all_metrics.append(int(arch))

    for i in range(len(arch_with_all_metrics)):
        for j in range(len(arch_with_all_metrics)):

            if arch_with_all_metrics[i] == arch_with_all_metrics[j]:
                continue

            if arch_with_all_metrics[i] < arch_with_all_metrics[j]:
                ele = str(arch_with_all_metrics[i]) + "__" + str(arch_with_all_metrics[j])
            else:
                ele = str(arch_with_all_metrics[j]) + "__" + str(arch_with_all_metrics[i])
            if ele in visited:
                continue

            num_dist += 1
            visited[ele] = 1

        num_log_index += 1
        if num_log_index % num_log == 0:
            print("count ", num_log_index)

    print(num_dist)
    print(len(visited.keys()))

    for i, ele in enumerate(np.array_split(list(visited.keys()), 8)):

        new_dict = {}
        for key in list(ele):
            new_dict[key] = 1

        with open(output_file + "/partition-" + str(i), 'w') as outfile:
            outfile.write(json.dumps(new_dict))
        del new_dict


def measure_correlation_all(data, dataset):
    """

    :param data:
    :param gt_api: validation_accuracy or test_accuracy
    :param rand_pick:
    :return:
    """
    print("---------------- begin to measure ----------------")

    id_list = []
    # score: {algName: [gt1, gt2...] }
    test_accuracy = {}
    # score: {algName: [s1, s2...] }
    scores = {}
    for arch_id, info in data.items():
        try:
            gt_test_acc, gt_train_time = gt_api_func(str(arch_id), dataset)
        except:
            continue # if this arch_id has not in id_to_hash, todo: need to update id-to-hash

        # record the arch_id
        id_list.append(arch_id)

        # record the score calculated by each algorithm for arch_id,
        for alg_name, score in info.items():
            f_score = float(score)
            if alg_name in scores:
                scores[alg_name].append(f_score)
            else:
                scores[alg_name] = []
                scores[alg_name].append(f_score)

            if alg_name in test_accuracy:
                test_accuracy[alg_name].append(gt_test_acc)
            else:
                test_accuracy[alg_name] = []
                test_accuracy[alg_name].append(gt_test_acc)

    # score: {algName: [s1, s2...] }
    correlation_result = {}
    for alg_name, score_list in scores.items():
        try:
            print(f"Compare on {len(scores[alg_name][:15625])} models")
            # get max 15625 models
            correlation_result[alg_name] = CorCoefficient.measure(scores[alg_name][:15625], test_accuracy[alg_name][:15625])
        except Exception as e:
            print(alg_name + " has error", e)
    print("=============================================")
    alg_name_list = ["grad_norm", "grad_plain", "jacob_conv", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
            "fisher", "grasp", "snip", "synflow", "weight_norm", "['nas_wot', 'synflow']"]
    for alg_name in alg_name_list:
        if alg_name in correlation_result:
            print(alg_name, '%.2f' % (correlation_result[alg_name]["Spearman"]))
            # print(alg_name, '%.2f, %.2f, %.2f' % (correlation_result[alg_name]["Pearson"], correlation_result[alg_name]["KendallTau"], correlation_result[alg_name]["Spearman"]))

    return id_list


def measure_correlation(input_file, dataset):
    file_name = input_file.split("/")[-1]
    print("+++++ measure correlation with", file_name, "+++++")
    with open(input_file, 'r') as readfile1:
        data = json.load(readfile1)

    measure_correlation_all(data, dataset)


def visualize_acc_score_plot(union_best_file_path, gt_file, vote_res, ylims, window_size):
    with open(union_best_file_path, 'r') as readfile:
        data = json.load(readfile)

    with open(gt_file, 'r') as readfile2:
        gt = json.load(readfile2)

    with open(vote_res, 'r') as readfile3:
        votedata = json.load(readfile3)

    all_RUN_result = {}
    ground_truth = []

    for arch_id, info in data.items():
        for alg_name, score in info.items():
            f_score = float(score)
            if alg_name not in all_RUN_result:
                all_RUN_result[alg_name] = []
            all_RUN_result[alg_name].append(f_score)

        for vote_comb_name in ["['nas_wot', 'synflow']"]:
            if vote_comb_name not in all_RUN_result:
                all_RUN_result[vote_comb_name] = []
            all_RUN_result[vote_comb_name].append(float(votedata[arch_id][vote_comb_name]))

        ground_truth.append(float(gt[str(arch_id)][dataset]['test-accuracy']))

    exist_keys = list(all_RUN_result.keys())

    # 0. origin score & acc
    all_RUN_result_ori = copy.deepcopy(all_RUN_result)
    for algName in exist_keys:
        sorted_score, batched_gt = sort_update(all_RUN_result_ori[algName], ground_truth)
        all_RUN_result_ori[algName] = sorted_score
        all_RUN_result_ori[algName + "_gt"] = batched_gt
    draw_sampler_res_sub(all_RUN_result_ori, "ori.jpg")

    # 1. rank score & acc
    all_RUN_result_rank = copy.deepcopy(all_RUN_result)
    for algName in exist_keys:
        # draw with rank
        all_RUN_result_rank[algName] = get_rank_after_sort(all_RUN_result_rank[algName])
        sorted_score, batched_gt = sort_update(all_RUN_result_rank[algName], ground_truth)
        # batch by b samples
        all_RUN_result_rank[algName] = sorted_score
        all_RUN_result_rank[algName + "_gt"] = batched_gt

    draw_sampler_res_sub(all_RUN_result_rank, "rank.jpg")

    # 2. rank score bath avg & + acc
    all_RUN_result_rank_batch = copy.deepcopy(all_RUN_result)
    for algName in exist_keys:
        # draw with rank
        all_RUN_result_rank_batch[algName] = get_rank_after_sort(all_RUN_result_rank_batch[algName])

        # batch by b samples
        sorted_score, batched_gt = sort_update_with_batch_average(all_RUN_result_rank_batch[algName], ground_truth, window_size)
        all_RUN_result_rank_batch[algName] = sorted_score
        all_RUN_result_rank_batch[algName + "_gt"] = batched_gt

    draw_sampler_res_sub(all_RUN_result_rank_batch, "rank_avg.jpg", ylims)

    # 3. rank score bath avg & + acc
    all_RUN_result_rank_batch_plot = copy.deepcopy(all_RUN_result)
    for algName in exist_keys:
        # draw with rank
        all_RUN_result_rank_batch_plot[algName] = get_rank_after_sort(all_RUN_result_rank_batch_plot[algName])

        # batch by b samples
        sorted_score, batched_gt_high, batched_gt_mean, batched_gt_low = \
            sort_update_with_batch_average_hlm(
                all_RUN_result_rank_batch_plot[algName], ground_truth, window_size)
        all_RUN_result_rank_batch_plot[algName] = sorted_score
        all_RUN_result_rank_batch_plot[algName + "_gth"] = batched_gt_high
        all_RUN_result_rank_batch_plot[algName + "_gtl"] = batched_gt_low
        all_RUN_result_rank_batch_plot[algName + "_gtm"] = batched_gt_mean
    draw_sampler_res_sub_plot(all_RUN_result_rank_batch_plot, "rank_avg_plot.jpg", ylims)


def draw_sampler_res_sub_plot(all_RUN_result, imageName, ylim=[]):
    # define plit function
    def plot_experiment(scores, label, axsp, high_acc, low_acc, mena_acc):

        axsp.plot(mena_acc)
        axsp.fill_between(range(len(scores)), low_acc, high_acc, alpha=0.3)
        axsp.set_title(label, fontsize=10)
        if len(ylim) > 0:
            axsp.set_ylim([ylim[0], ylim[1]])

    f, allaxs = plt.subplots(2, 6, sharey="row", figsize=(15, 9))
    allaxs = allaxs.ravel()
    index = 0

    keys = ["grad_norm", "grad_plain", "nas_wot", "grasp", "synflow",
            "ntk_trace", "fisher", "weight_norm", "ntk_cond_num", "snip", "ntk_trace_approx", "['nas_wot', 'synflow']"]
    for algname in keys:
        info = all_RUN_result[algname]
        high_acc = all_RUN_result[algname + "_gth"]
        low_acc = all_RUN_result[algname + "_gtl"]
        mena_acc = all_RUN_result[algname + "_gtm"]
        if algname == "ntk_cond_num":
            plot_experiment(info, "ntk_cond", allaxs[index], high_acc, low_acc, mena_acc)
        else:
            plot_experiment(info, algname, allaxs[index], high_acc, low_acc, mena_acc)
        index += 1

    # f.delaxes(allaxs[11])
    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    plt.tight_layout()
    plt.show()
    f.savefig(imageName, bbox_inches='tight')


def draw_sampler_res_sub(all_RUN_result, imageName, ylim=[]):
    # define plit function
    def plot_experiment(scores, label, axsp, acc_m):
        axsp.scatter(scores, acc_m)
        # axsp.set_xticks([])
        axsp.set_title(label, fontsize=10)
        # axsp.grid()
        if len(ylim) > 0:
            axsp.set_ylim([ylim[0], ylim[1]])

    f, allaxs = plt.subplots(2, 6, sharey="row", figsize=(15, 9))
    allaxs = allaxs.ravel()
    index = 0

    keys = ["grad_norm", "grad_plain", "nas_wot", "grasp", "synflow",
            "ntk_trace", "fisher", "weight_norm", "ntk_cond_num", "snip", "ntk_trace_approx", "['nas_wot', 'synflow']"]
    for algname in keys:
        info = all_RUN_result[algname]
        acc = all_RUN_result[algname + "_gt"]
        if algname == "ntk_cond_num":
            plot_experiment(info, "ntk_cond", allaxs[index], acc)
        else:
            plot_experiment(info, algname, allaxs[index], acc)
        index += 1

    # f.delaxes(allaxs[11])
    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    plt.tight_layout()
    plt.show()
    f.savefig(imageName, bbox_inches='tight')


if __name__ == "__main__":
    """
    This scripts will
        1. combine BN and noBN's synflow results
        2. measure the SRCC
        3. convert to rank and add vote result, then measure the SRCC
    Switching between 12 and 200 epoch in 201: change gt_api_func
    """

    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "result_base", "result_append", "CIFAR10_15625")
    output_dir = os.path.join(cwd, "result_base", "result_append", "CIFAR10_15625", "union")

    # this is NB201+ C10
    # per_fix = "201_15625_c10_bs32_ic16"
    # dataset = "cifar10"
    # gt_api = Gt201()
    # gt_api_func = gt_api.query_12_epoch

    # this is NB101+ C10
    per_fix = "101_15k_c10_bs32_ic16"
    dataset = "cifar10"
    gt_api = Gt101()
    gt_api_func = gt_api.get_c10_test_info

    bn_input_file_path = os.path.join(data_dir, per_fix + "_BN.json")
    noBn_input_file_path = os.path.join(data_dir, per_fix + "_noBN.json")
    # measure_correlation(bn_input_file_path, dataset)
    # measure_correlation(noBn_input_file_path, dataset)

    # 1. union bn and no-bn result
    union_best_file_save_path = os.path.join(output_dir, per_fix + "_unionBest.json")
    union_best_bn_cfg(bn_input_file_path, noBn_input_file_path, union_best_file_save_path)
    print("-----stage-1 union bn and no-bn synflow done-----")
    # 1.1. after union, measure correlation
    measure_correlation(union_best_file_save_path, dataset)
    print("-----stage-1 measure correlation done-----")

    # 2. convert to score and update rank
    vote_file_save_path = os.path.join(output_dir, per_fix + "_unionBest_with_vote.json")
    add_vote_info(union_best_file_save_path, vote_file_save_path)
    # 2.1 measure rank correlation
    measure_correlation(vote_file_save_path, dataset)
    print("-----stage-2 convert-2-score, and measure correlation done-----")

    # 3. visualize acc score
    # visualize_acc_score_plot(union_best_file_path, gt_file, vote_file_path, [0.8, 0.95], 50)

    # 4. partition key into 8 groups
    # partition_key_int_groups(vote_file_path, output_dir)

    # 5. vote using golang








