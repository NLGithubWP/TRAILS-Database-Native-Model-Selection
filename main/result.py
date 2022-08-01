
import json
from statistic_lib import parse_dict_measure


with open('./Logs/cifar10_15000/101_30k_c10_128_noBN.json', 'r') as readfile:
    data = json.load(readfile)


com_num = len(data.keys())
id_ls1 = parse_dict_measure(data, "test_accuracy", com_num)

