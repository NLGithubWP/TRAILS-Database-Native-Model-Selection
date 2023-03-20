import os.path

from utilslibs.io_tools import read_json, write_json

base_dir = "./base_line_res_2k5/"
all_files = os.listdir(base_dir)

result = {}

dataset = ""
for each_worker_info in all_files:
    each_dic = read_json(os.path.join(base_dir, each_worker_info))
    if dataset == "":
        dataset = list(each_dic.keys())[0]
    if dataset not in result:
        result[dataset] = {}
    result[dataset].update(each_dic[dataset])

write_json(base_dir+"all_train_baseline_2k5.json", result)
