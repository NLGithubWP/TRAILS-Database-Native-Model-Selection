import json
import numpy as np




with open('//result/CIFAR10_15625/vote_res/101_15625_c10_128_unionBest_withvote.json', 'r') as readfile:
    data = json.load(readfile)

visited = {}
num_dist = 0

num_log = 1000
num_log_index = 0

aa = list(data.keys())
all_ss = [
    "grad_norm", "grad_plain", "nas_wot", "ntk_cond_num", "ntk_trace",
    "ntk_trace_approx", "fisher", "grasp", "snip", "synflow"]

a = []
for arch in aa:
    isContinue = 1
    for ele in all_ss:
        if ele not in data[arch]["scores"]:
            isContinue = 0
            break
    if isContinue == 0:
        continue
    a.append(arch)

for i in range(len(a)):
    for j in range(len(a)):

        if a[i] == a[j]:
            continue

        if a[i] < a[j]:
            ele = str(a[i]) + "__" + str(a[j])
        else:
            ele = str(a[j]) + "__" + str(a[i])
        if ele in visited:
            continue

        num_dist += 1
        visited[ele] = 1
    num_log_index += 1
    if num_log_index % num_log == 0:
        print("count ", num_log_index)

print(num_dist)
print(len(visited.keys()))


for i, ele in enumerate(np.array_split(list(visited.keys()), 8)):

    new_dict = {}
    for key in list(ele):
        new_dict[key] = 1

    with open("/Users/kevin/project_python/FIRMEST/result/CIFAR10_15625/101_correct_pair/partition-"+str(i), 'w') as outfile:
        outfile.write(json.dumps(new_dict))
    del new_dict


