


import json
import math

file_name = "//result/CIFAR10_15625/vote_res/201_15625_c10_128_unionBest.json"
with open(file_name, 'r') as readfile:
    dataBest = json.load(readfile)

# dataBest in terms of
"""
{arch_id: {test_accuracy: 0.5, scores : {nas_wot : {score: 0.5, gpu_time: 0.5 }}} }
"""

# get one arch_id's information
for arch_id in dataBest:
    acc = float(dataBest[arch_id]["test_accuracy"])
    print(arch_id, "test accuracy is", float(acc))
    for alg_name in dataBest[arch_id]["scores"]:
        score = float(dataBest[arch_id]["scores"][alg_name]["score"])
        if math.isinf(score):
            if score > 0:
                score = 1e8
            else:
                score = -1e8
        print(arch_id, "score evaluated on", alg_name, "is", score)

    break
