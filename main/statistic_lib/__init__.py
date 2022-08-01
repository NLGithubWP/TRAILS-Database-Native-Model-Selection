


import statistics
from pprint import pprint
import math
from measure.correlation_coefficient import CorCoefficient


def parse_dict_measure(data, accuracy_type: str, rand_pick: int):
    """

    :param data:
    :param accuracy_type: validation_accuracy or test_accuracy
    :param rand_pick:
    :return:
    """
    print("---------------- begin to measure ----------------")

    id_list = []

    test_accuracy = []

    # score: {algName: [s1, s2...] }
    scores = {}
    for arch_id, info in data.items():
        if accuracy_type not in info:
            continue
        test_accuracy.append(info[accuracy_type])
        id_list.append(arch_id)

        for alg_name, score_info in info["scores"].items():
            if alg_name in scores:
                scores[alg_name].append(score_info["score"])
            else:
                scores[alg_name] = []
                scores[alg_name].append(score_info["score"])

    # score: {algName: [s1, s2...] }
    res = {}
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
            res[alg_name] = CorCoefficient.measure(picked_score[:rand_pick], picked_accuracy[:rand_pick])
            print(alg_name + " is measured on arch = ", len(picked_score))
        except Exception as e:
            print(alg_name + " has error", e)
    print("=============================================")
    list = ["grad_norm", "grad_plain", "jacob_conv", "nas_wot", "ntk_cond_num", "ntk_trace", "ntk_trace_approx",
            "fisher", "grasp", "snip", "synflow", "weight_norm"]
    for key in list:
        if key in res:
            print(key, '%.4f, %.4f, %.4f' % (res[key]["Pearson"], res[key]["KendallTau"], res[key]["Spearman"]))

    return id_list[:rand_pick]


def get_rank_after_sort(a: list):
    """
    Get rank after sorting, [3,1,2] => [2,0,1]
    :param a:
    :return:
    """
    mapper = {}
    asort = sorted(a)
    for i, ele in enumerate(asort):
        mapper[ele] = i
    res = []

    for ele in a:
        res.append(mapper[ele])
    return res


def sort_update(x, y):
    """
    Sort y according to x
    :param x:
    :param y:
    :return:
    """
    y = [ele for _, ele in sorted(zip(x, y))]
    x = sorted(x)
    new_x = []
    index = 0
    index_new_x = 0
    new_y = []
    b = 30
    while index < len(x):
        new_x.append(index_new_x)
        if len(y[index: index+b]) == 0:
            print("err")
        new_y.append(statistics.mean(y[index: index+b]))
        index += b
        index_new_x += 1

    return new_x, new_y


def print_one_example(data):
    key_list = []
    for k, v in data.items():
        pprint(k)
        pprint(v)
        key_list.append(v["architecture_id"])
        break
    print(key_list)


