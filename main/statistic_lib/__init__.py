


import statistics
from pprint import pprint
import math

import numpy as np

from measure.correlation_coefficient import CorCoefficient


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
    :param y: acc
    :param x: scores list
    :return:
    """
    y = [ele for _, ele in sorted(zip(x, y))]
    x = sorted(x)
    return x, y


def sort_update_with_batch_average(x, y, b):
    """
    Sort y according to x
    :param x: scores list
    :param y: accuracy
    :return:
    """
    y = [ele for _, ele in sorted(zip(x, y))]
    x = sorted(x)
    new_x = []
    index = 0
    index_new_x = 0
    new_y = []
    while index < len(x):
        new_x.append(index_new_x)
        if len(y[index: index+b]) == 0:
            print("err")
        new_y.append(statistics.mean(y[index: index+b]))
        index += b
        index_new_x += 1
    return new_x, new_y

def sort_update_with_batch_average_hlm(x, y, b):
    """
    Sort y according to x
    :param x: scores list
    :param y: accuracy
    :return:
    """
    y = [ele for _, ele in sorted(zip(x, y))]
    x = sorted(x)
    new_x = []
    index = 0
    index_new_x = 0
    new_y_high = []
    new_y_low = []
    new_y_mean = []

    while index < len(x):
        new_x.append(index_new_x)
        if len(y[index: index+b]) == 0:
            print("err")
        new_y_mean.append(statistics.mean(y[index: index+b]))
        new_y_high.append(np.percentile(y[index: index + b], 75))
        new_y_low.append(np.percentile(y[index: index + b], 25))
        # new_y_high.append(max(y[index: index + b]))
        # new_y_low.append(min(y[index: index + b]))

        index += b
        index_new_x += 1
    return new_x, new_y_high, new_y_mean, new_y_low


def print_one_example(data):
    key_list = []
    for k, v in data.items():
        pprint(k)
        pprint(v)
        key_list.append(v["architecture_id"])
        break
    print(key_list)


