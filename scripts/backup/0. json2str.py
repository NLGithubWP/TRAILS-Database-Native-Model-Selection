


import json
import os

cwd = os.getcwd()
data_dir = os.path.join(cwd, "result_sensitive")


for file_name in os.listdir(os.path.join(data_dir, "ori")):

    input_file_path = os.path.join(data_dir, "ori", file_name)
    output_file_path = os.path.join(data_dir, "201_CIFAR10_15625", file_name)

    with open(input_file_path, 'r') as readfile:
        data = json.load(readfile)

    new_data = {}

    for arch_id in data.keys():
        new_data[arch_id] = data[arch_id]
        for alg_name, info in new_data[arch_id].items():
            new_data[arch_id][alg_name] = '{:f}'.format(new_data[arch_id][alg_name])

    with open(output_file_path, 'w') as outfile:
        outfile.write(json.dumps(new_data))
