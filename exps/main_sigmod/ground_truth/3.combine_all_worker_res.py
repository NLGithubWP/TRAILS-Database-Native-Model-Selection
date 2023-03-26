import os.path

from utilslibs.io_tools import read_json, write_json
import os.path


# fetch result from server. rename base_line_res to base_line_res_2k5

base_dir = "./A_structure_dataexp_res/uci/20230327_checkpointing/saved_result"
all_files = os.listdir(base_dir)
output_folder = os.path.dirname(base_dir)

result = {}

dataset = ""
for each_worker_info in all_files:
    each_dic = read_json(os.path.join(base_dir, each_worker_info))
    if dataset == "":
        dataset = list(each_dic.keys())[0]
    if dataset not in result:
        result[dataset] = {}
    result[dataset].update(each_dic[dataset])

write_json(os.path.join(output_folder, "all_train_baseline_uci.json"), result)
