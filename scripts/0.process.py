import copy
import json
import os
import scipy.stats as ss
import numpy as np
from os.path import exists

from matplotlib import pyplot as plt

from measure.correlation_coefficient import CorCoefficient
from statistic_lib import get_rank_after_sort, sort_update, sort_update_with_batch_average, \
    sort_update_with_batch_average_hlm


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
            res.append(float(info[algName]))
            archIds.append(archId)
        ranked_res = ss.rankdata(res)

        # update origin dict
        for archId, info in data.items():
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

    for metrics in all_ss:
        update_score_to_rank(data, metrics)

    all_keys = data.keys()
    print(len(all_keys))

    # total_vote_combination = [["grad_norm", "grad_plain", ], ["grad_norm", "nas_wot", ], ["grad_plain", "nas_wot", ],
    #                           ["grad_norm", "ntk_cond_num", ], ["grad_plain", "ntk_cond_num", ],
    #                           ["nas_wot", "ntk_cond_num", ], ["grad_norm", "ntk_trace", ],
    #                           ["grad_plain", "ntk_trace", ],
    #                           ["nas_wot", "ntk_trace", ], ["ntk_cond_num", "ntk_trace", ],
    #                           ["grad_norm", "ntk_trace_approx", ], ["grad_plain", "ntk_trace_approx", ],
    #                           ["nas_wot", "ntk_trace_approx", ], ["ntk_cond_num", "ntk_trace_approx", ],
    #                           ["ntk_trace", "ntk_trace_approx", ], ["grad_norm", "fisher", ],
    #                           ["grad_plain", "fisher", ],
    #                           ["nas_wot", "fisher", ], ["ntk_cond_num", "fisher", ], ["ntk_trace", "fisher", ],
    #                           ["ntk_trace_approx", "fisher", ], ["grad_norm", "grasp", ], ["grad_plain", "grasp", ],
    #                           ["nas_wot", "grasp", ], ["ntk_cond_num", "grasp", ], ["ntk_trace", "grasp", ],
    #                           ["ntk_trace_approx", "grasp", ], ["fisher", "grasp", ], ["grad_norm", "snip", ],
    #                           ["grad_plain", "snip", ], ["nas_wot", "snip", ], ["ntk_cond_num", "snip", ],
    #                           ["ntk_trace", "snip", ], ["ntk_trace_approx", "snip", ], ["fisher", "snip", ],
    #                           ["grasp", "snip", ], ["grad_norm", "synflow", ], ["grad_plain", "synflow", ],
    #                           ["nas_wot", "synflow", ], ["ntk_cond_num", "synflow", ], ["ntk_trace", "synflow", ],
    #                           ["ntk_trace_approx", "synflow", ], ["fisher", "synflow", ], ["grasp", "synflow", ],
    #                           ["snip", "synflow", ], ["grad_norm", "grad_plain", "nas_wot", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", ], ["grad_norm", "nas_wot", "ntk_cond_num", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", ], ["grad_norm", "grad_plain", "ntk_trace", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", ], ["grad_plain", "nas_wot", "ntk_trace", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", ],
    #                           ["grad_norm", "grad_plain", "fisher", ],
    #                           ["grad_norm", "nas_wot", "fisher", ], ["grad_plain", "nas_wot", "fisher", ],
    #                           ["grad_norm", "ntk_cond_num", "fisher", ], ["grad_plain", "ntk_cond_num", "fisher", ],
    #                           ["nas_wot", "ntk_cond_num", "fisher", ], ["grad_norm", "ntk_trace", "fisher", ],
    #                           ["grad_plain", "ntk_trace", "fisher", ], ["nas_wot", "ntk_trace", "fisher", ],
    #                           ["ntk_cond_num", "ntk_trace", "fisher", ], ["grad_norm", "ntk_trace_approx", "fisher", ],
    #                           ["grad_plain", "ntk_trace_approx", "fisher", ],
    #                           ["nas_wot", "ntk_trace_approx", "fisher", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "fisher", ],
    #                           ["ntk_trace", "ntk_trace_approx", "fisher", ], ["grad_norm", "grad_plain", "grasp", ],
    #                           ["grad_norm", "nas_wot", "grasp", ], ["grad_plain", "nas_wot", "grasp", ],
    #                           ["grad_norm", "ntk_cond_num", "grasp", ], ["grad_plain", "ntk_cond_num", "grasp", ],
    #                           ["nas_wot", "ntk_cond_num", "grasp", ], ["grad_norm", "ntk_trace", "grasp", ],
    #                           ["grad_plain", "ntk_trace", "grasp", ], ["nas_wot", "ntk_trace", "grasp", ],
    #                           ["ntk_cond_num", "ntk_trace", "grasp", ], ["grad_norm", "ntk_trace_approx", "grasp", ],
    #                           ["grad_plain", "ntk_trace_approx", "grasp", ], ["nas_wot", "ntk_trace_approx", "grasp", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "grasp", ],
    #                           ["ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "fisher", "grasp", ], ["grad_plain", "fisher", "grasp", ],
    #                           ["nas_wot", "fisher", "grasp", ], ["ntk_cond_num", "fisher", "grasp", ],
    #                           ["ntk_trace", "fisher", "grasp", ], ["ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "snip", ], ["grad_norm", "nas_wot", "snip", ],
    #                           ["grad_plain", "nas_wot", "snip", ], ["grad_norm", "ntk_cond_num", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "snip", ], ["nas_wot", "ntk_cond_num", "snip", ],
    #                           ["grad_norm", "ntk_trace", "snip", ], ["grad_plain", "ntk_trace", "snip", ],
    #                           ["nas_wot", "ntk_trace", "snip", ], ["ntk_cond_num", "ntk_trace", "snip", ],
    #                           ["grad_norm", "ntk_trace_approx", "snip", ], ["grad_plain", "ntk_trace_approx", "snip", ],
    #                           ["nas_wot", "ntk_trace_approx", "snip", ], ["ntk_cond_num", "ntk_trace_approx", "snip", ],
    #                           ["ntk_trace", "ntk_trace_approx", "snip", ], ["grad_norm", "fisher", "snip", ],
    #                           ["grad_plain", "fisher", "snip", ], ["nas_wot", "fisher", "snip", ],
    #                           ["ntk_cond_num", "fisher", "snip", ], ["ntk_trace", "fisher", "snip", ],
    #                           ["ntk_trace_approx", "fisher", "snip", ], ["grad_norm", "grasp", "snip", ],
    #                           ["grad_plain", "grasp", "snip", ], ["nas_wot", "grasp", "snip", ],
    #                           ["ntk_cond_num", "grasp", "snip", ], ["ntk_trace", "grasp", "snip", ],
    #                           ["ntk_trace_approx", "grasp", "snip", ], ["fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "synflow", ], ["grad_norm", "nas_wot", "synflow", ],
    #                           ["grad_plain", "nas_wot", "synflow", ], ["grad_norm", "ntk_cond_num", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "synflow", ], ["nas_wot", "ntk_cond_num", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "synflow", ], ["grad_plain", "ntk_trace", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "synflow", ], ["ntk_cond_num", "ntk_trace", "synflow", ],
    #                           ["grad_norm", "ntk_trace_approx", "synflow", ],
    #                           ["grad_plain", "ntk_trace_approx", "synflow", ],
    #                           ["nas_wot", "ntk_trace_approx", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "synflow", ],
    #                           ["ntk_trace", "ntk_trace_approx", "synflow", ], ["grad_norm", "fisher", "synflow", ],
    #                           ["grad_plain", "fisher", "synflow", ], ["nas_wot", "fisher", "synflow", ],
    #                           ["ntk_cond_num", "fisher", "synflow", ], ["ntk_trace", "fisher", "synflow", ],
    #                           ["ntk_trace_approx", "fisher", "synflow", ], ["grad_norm", "grasp", "synflow", ],
    #                           ["grad_plain", "grasp", "synflow", ], ["nas_wot", "grasp", "synflow", ],
    #                           ["ntk_cond_num", "grasp", "synflow", ], ["ntk_trace", "grasp", "synflow", ],
    #                           ["ntk_trace_approx", "grasp", "synflow", ], ["fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "snip", "synflow", ], ["grad_plain", "snip", "synflow", ],
    #                           ["nas_wot", "snip", "synflow", ], ["ntk_cond_num", "snip", "synflow", ],
    #                           ["ntk_trace", "snip", "synflow", ], ["ntk_trace_approx", "snip", "synflow", ],
    #                           ["fisher", "snip", "synflow", ], ["grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "fisher", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "fisher", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "fisher", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "fisher", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "fisher", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "fisher", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "fisher", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "fisher", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "fisher", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "fisher", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "fisher", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "fisher", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "grasp", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "grasp", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "grasp", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "grasp", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "grasp", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "grad_plain", "fisher", "grasp", ],
    #                           ["grad_norm", "nas_wot", "fisher", "grasp", ],
    #                           ["grad_plain", "nas_wot", "fisher", "grasp", ],
    #                           ["grad_norm", "ntk_cond_num", "fisher", "grasp", ],
    #                           ["grad_plain", "ntk_cond_num", "fisher", "grasp", ],
    #                           ["nas_wot", "ntk_cond_num", "fisher", "grasp", ],
    #                           ["grad_norm", "ntk_trace", "fisher", "grasp", ],
    #                           ["grad_plain", "ntk_trace", "fisher", "grasp", ],
    #                           ["nas_wot", "ntk_trace", "fisher", "grasp", ],
    #                           ["ntk_cond_num", "ntk_trace", "fisher", "grasp", ],
    #                           ["grad_norm", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_plain", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["nas_wot", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["ntk_trace", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "grad_plain", "fisher", "snip", ],
    #                           ["grad_norm", "nas_wot", "fisher", "snip", ],
    #                           ["grad_plain", "nas_wot", "fisher", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "fisher", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "fisher", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "fisher", "snip", ],
    #                           ["grad_norm", "ntk_trace", "fisher", "snip", ],
    #                           ["grad_plain", "ntk_trace", "fisher", "snip", ],
    #                           ["nas_wot", "ntk_trace", "fisher", "snip", ],
    #                           ["ntk_cond_num", "ntk_trace", "fisher", "snip", ],
    #                           ["grad_norm", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_plain", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["nas_wot", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["ntk_trace", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_trace", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_trace", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_trace", "grasp", "snip", ],
    #                           ["ntk_cond_num", "ntk_trace", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["ntk_trace", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_norm", "fisher", "grasp", "snip", ], ["grad_plain", "fisher", "grasp", "snip", ],
    #                           ["nas_wot", "fisher", "grasp", "snip", ], ["ntk_cond_num", "fisher", "grasp", "snip", ],
    #                           ["ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "grad_plain", "fisher", "synflow", ],
    #                           ["grad_norm", "nas_wot", "fisher", "synflow", ],
    #                           ["grad_plain", "nas_wot", "fisher", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "fisher", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "fisher", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "fisher", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "fisher", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "fisher", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "fisher", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "fisher", "synflow", ],
    #                           ["grad_norm", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_plain", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["nas_wot", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["ntk_trace", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_norm", "grad_plain", "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "grasp", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["ntk_trace", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_norm", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "fisher", "grasp", "synflow", ],
    #                           ["nas_wot", "fisher", "grasp", "synflow", ],
    #                           ["ntk_cond_num", "fisher", "grasp", "synflow", ],
    #                           ["ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "snip", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["ntk_trace", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_norm", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "fisher", "snip", "synflow", ],
    #                           ["nas_wot", "fisher", "snip", "synflow", ],
    #                           ["ntk_cond_num", "fisher", "snip", "synflow", ],
    #                           ["ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "grasp", "snip", "synflow", ], ["grad_plain", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "grasp", "snip", "synflow", ], ["ntk_cond_num", "grasp", "snip", "synflow", ],
    #                           ["ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "fisher", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "fisher", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "fisher", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "fisher", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "fisher", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "fisher", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "fisher", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "fisher", "grasp", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "fisher", "grasp", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "grasp", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "fisher", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "fisher", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "fisher", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "fisher", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "fisher", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "fisher", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "fisher", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "fisher", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "fisher", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "fisher", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "fisher", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "fisher", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "fisher", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "fisher", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "fisher", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "fisher", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "fisher", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "fisher", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "fisher", "grasp", "snip", "synflow", ],
    #                           ["ntk_cond_num", "fisher", "grasp", "snip", "synflow", ],
    #                           ["ntk_trace", "fisher", "grasp", "snip", "synflow", ],
    #                           ["ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "fisher", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "fisher", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "fisher", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "fisher", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "fisher", "grasp", "snip", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "fisher", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher",
    #                            "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher",
    #                            "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp",
    #                            "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp",
    #                            "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp",
    #                            "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp",
    #                            "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp",
    #                            "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp",
    #                            "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp",
    #                            "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp",
    #                            "snip", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp",
    #                            "snip", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp",
    #                            "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp",
    #                            "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp",
    #                            "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp",
    #                            "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip",
    #                            "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip",
    #                            "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip",
    #                            "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip",
    #                            "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip",
    #                            "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "fisher", "grasp", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "fisher", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp",
    #                            "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher",
    #                            "grasp",
    #                            "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp",
    #                            "snip", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp", "snip", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp",
    #                            "snip", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp",
    #                            "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "fisher", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher",
    #                            "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp",
    #                            "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher",
    #                            "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "snip",
    #                            "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "grasp",
    #                            "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp",
    #                            "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "grasp",
    #                            "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace_approx", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher", "grasp", "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "fisher", "grasp", "snip", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "fisher", "grasp", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "fisher", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
    #                            "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "fisher", "grasp",
    #                            "snip",
    #                            "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace_approx", "fisher",
    #                            "grasp",
    #                            "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "nas_wot", "ntk_trace", "ntk_trace_approx", "fisher", "grasp",
    #                            "snip", "synflow", ],
    #                           ["grad_norm", "grad_plain", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp", "snip", "synflow", ],
    #                           ["grad_norm", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp",
    #                            "snip", "synflow", ],
    #                           ["grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx", "fisher",
    #                            "grasp",
    #                            "snip", "synflow", ],
    #                           ]

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


def measure_correlation_all(data, space, dataset):
    """

    :param data:
    :param space: validation_accuracy or test_accuracy
    :param rand_pick:
    :return:
    """
    print("---------------- begin to measure ----------------")

    id_list = []
    test_accuracy = []
    # score: {algName: [s1, s2...] }
    scores = {}
    for arch_id, info in data.items():
        id_list.append(arch_id)
        for alg_name, score in info.items():
            f_score = float(score)
            if alg_name in scores:
                scores[alg_name].append(f_score)
            else:
                scores[alg_name] = []
                scores[alg_name].append(f_score)

        query_info = space[str(arch_id)][dataset]
        test_accuracy.append(float(query_info['test-accuracy']))

    # score: {algName: [s1, s2...] }
    res = {}
    for alg_name, score_list in scores.items():
        try:
            res[alg_name] = CorCoefficient.measure(scores[alg_name], test_accuracy)
        except Exception as e:
            print(alg_name + " has error", e)
    print("=============================================")
    list = ["grad_norm", "grad_plain", "jacob_conv", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
            "fisher", "grasp", "snip", "synflow", "weight_norm", "['nas_wot', 'synflow']"]
    for key in list:
        if key in res:
            print(key, '%.2f, %.2f, %.2f' % (res[key]["Pearson"], res[key]["KendallTau"], res[key]["Spearman"]))

    return id_list


def measure_correlation(input_file, gt_file, dataset):
    file_name = input_file.split("/")[-1]
    print("+++++ measure correlation with", file_name, "+++++")
    with open(input_file, 'r') as readfile1:
        data = json.load(readfile1)

    with open(gt_file, 'r') as readfile2:
        gt = json.load(readfile2)

    measure_correlation_all(data, gt, dataset)


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
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "result_append", "CIFAR10_15625")
    output_dir = os.path.join(data_dir, "union")
    gt_file = os.path.join(cwd, "201_200_result")

    per_fix = "201_15625_c10_bs32_ic16"
    dataset = "cifar10"

    # per_fix = "201_15k_imgNet_bs32_ic16"
    # dataset = "ImageNet16-120"
    # per_fix = "201_15k_c10_bs32_ic16"
    # dataset = "cifar10"
    # per_fix = "201_15k_c100_bs32_ic16"
    # dataset = "cifar100"

    bn_input_file_path = os.path.join(output_dir, per_fix + "_BN.json")
    noBn_input_file_path = os.path.join(output_dir, per_fix + "_noBN.json")

    # 1. union bn and no-bn result
    union_best_file_path = os.path.join(output_dir, per_fix + "_unionBest.json")
    union_best_bn_cfg(bn_input_file_path, noBn_input_file_path, union_best_file_path)
    print("-----stage-1 union bn and no-bn synflow done-----")
    # 1.1. after union, measure correlation
    measure_correlation(union_best_file_path, gt_file, dataset)
    print("-----stage-2 measure correlation done-----")

    # 2. convert to score and update rank
    vote_file_path = os.path.join(output_dir, per_fix + "_unionBest_with_vote.json")
    add_vote_info(union_best_file_path, vote_file_path)
    # 2.1 measure rank correlation
    measure_correlation(vote_file_path, gt_file, dataset)
    print("-----stage-3 convert-2-score, and measure correlation done-----")

    # 3. visualize acc score
    # visualize_acc_score_plot(union_best_file_path, gt_file, vote_file_path, [0.8, 0.95], 50)

    # 4. partition key into 8 groups
    # partition_key_int_groups(vote_file_path, output_dir)

    # 5. vote using golang








