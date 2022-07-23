




import statistics
from pprint import pprint


def get_rank_after_sort(a: list):
    mapper = {}
    asort = sorted(a)
    for i, ele in enumerate(asort):
        mapper[ele] = i
    res = []

    for ele in a:
        res.append(mapper[ele])
    return res


def sort_update(x, y):
    y = [x for _, x in sorted(zip(x, y))]
    x = sorted(x)
    new_x = []
    index = 0
    index_new_x = 0
    new_y = []
    b = 100
    while index < len(x):
        new_x.append(index_new_x)
        if len(y[index: index+b]) == 0:
            print("err")
        new_y.append(statistics.mean(y[index: index+b]))
        index += b
        index_new_x += 1

    return x, y


def print_one_example(data):
    key_list = []
    for k, v in data.items():
        pprint(k)
        pprint(v)
        key_list.append(v["architecture_id"])
        break
    print(key_list)


