
import json
from src.measure.correlation_coefficient import CorCoefficient
from statistic_lib import get_rank_after_sort

with open('./Logs/101_15k_c10_128_noBN.json', 'r') as readfile:
    data = json.load(readfile)

test_accuracy = []
scores = {}

for arch_id, info in data.items():
    if "test_accuracy" not in info:
        continue
    test_accuracy.append(info["test_accuracy"])

    for alg_name, score_info in info["scores"].items():
        if alg_name in scores:
            scores[alg_name].append(score_info["score"])
        else:
            scores[alg_name] = []
            scores[alg_name].append(score_info["score"])

res = {}
import math
for alg_name, score_list in scores.items():

    picked_score = []
    picked_accuracy = []

    for i in range(len(score_list)):
        ele = score_list[i]
        if math.isnan(ele) or math.isinf(ele):
            continue
        picked_score.append(ele)
        picked_accuracy.append(test_accuracy[i])

    # picked_score = get_rank_after_sort(picked_score)
    # picked_accuracy = get_rank_after_sort(picked_accuracy)

    try:
        res[alg_name] = CorCoefficient.measure(picked_score, picked_accuracy)
        print(alg_name + " is measured on arch = ", len(picked_score))
    except Exception as e:
        print(alg_name + " has error", e)

# pprint(res)

print("=============================================")

list = ["grad_norm", "grad_plain", "jacob_conv", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
        "fisher", "grasp", "snip", "synflow", "weight_norm"]
for key in list:
    if key in res:
        # print(key, res[key]["Pearson"], res[key]["KendallTau"], res[key]["Spearman"] )
        print(key, '%.2f, %.2f, %.2f' % (res[key]["Pearson"], res[key]["KendallTau"], res[key]["Spearman"]))
