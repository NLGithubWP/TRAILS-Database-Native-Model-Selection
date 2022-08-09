

import json
import os

cwd = os.getcwd()
data_dir = os.path.join(cwd, "result_sensitive", "201_CIFAR10_15625")
bn_input_file_path = os.path.join(data_dir, "101_15625_c100_128_BN.json")
noBn_input_file_path = os.path.join(data_dir, "101_15625_c100_128_noBN.json")
output_file_path = os.path.join(data_dir, "union", "101_15625_c100_128_unionBest.json")


with open(bn_input_file_path, 'r') as readfile:
    data_no_bn = json.load(readfile)

with open(noBn_input_file_path, 'r') as readfile:
    data_bn = json.load(readfile)

new_data = {}

all_keys = set(data_no_bn.keys()).intersection( set(data_bn.keys()) )
print(len(all_keys), len(data_no_bn.keys()), len(data_bn.keys()))
for arch_id in list(all_keys):
    bn_info = data_bn[arch_id]
    no_bn_info = data_no_bn[arch_id]
    new_data[arch_id] = bn_info
    for alg_name in new_data[arch_id]:
        if alg_name == "synflow":
            new_data[arch_id][alg_name] = no_bn_info[arch_id][alg_name]
        else:
            new_data[arch_id][alg_name] = bn_info[arch_id][alg_name]

with open(output_file_path, 'w') as outfile:
    outfile.write(json.dumps(new_data))
