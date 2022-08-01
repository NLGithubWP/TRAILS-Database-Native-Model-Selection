import json

with open('./Logs/cifar10/201_15k_c10_128.json', 'r') as readfile:
    data = json.load(readfile)

with open('./Logs/cifar10/201_15k_c10_128_ntk.json', 'r') as readfile:
    data_ntk = json.load(readfile)

with open('./Logs/cifar10/201_15k_c10_128_synflow_noBN.json', 'r') as readfile:
    data_syflow = json.load(readfile)

for arch, info in data.items():
    if arch in data_ntk:
        for ntk_name, ntk_info in data_ntk[arch]["scores"].items():
            info["scores"][ntk_name] = ntk_info
    if arch in data_syflow:
        for syn_name, syn_info in data_syflow[arch]["scores"].items():
            info["scores"][syn_name] = syn_info


with open('./Logs/cifar10/201_15k_c10_128_all.json', 'w') as outfile:
    outfile.write(json.dumps(data))