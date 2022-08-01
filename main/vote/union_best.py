

import json

with open("/Users/kevin/project_python/Fast-AutoNAS/Logs/cifar10_15000/201_15k_c10_128_noBN.json", 'r') as readfile:
    data_no_bn = json.load(readfile)

with open("/Users/kevin/project_python/Fast-AutoNAS/Logs/cifar10_15000/201_15k_c10_128_BN.json", 'r') as readfile:
    data_bn = json.load(readfile)

new_data = {}


for ele in data_no_bn:
    if ele not in data_bn:
        print("1: ", ele)

for ele in data_bn:
    if ele not in data_no_bn:
        print("2: ", ele)

all_keys = set(data_no_bn.keys()).intersection( set(data_bn.keys()) )

print(len(all_keys))

for arch_id in list(all_keys):

    bn_info = data_bn[arch_id]
    no_bn_info = data_no_bn[arch_id]

    new_data[arch_id] = bn_info
    new_data[arch_id]['trainable_parameters'] = ""
    new_data[arch_id]['training_time'] = ""
    new_data[arch_id]['train_accuracy'] = str(new_data[arch_id]['train_accuracy'])
    new_data[arch_id]['validation_accuracy'] = str(new_data[arch_id]['validation_accuracy'])
    new_data[arch_id]['test_accuracy'] = str(new_data[arch_id]['test_accuracy'])
    new_data[arch_id]['architecture_id'] = str(new_data[arch_id]['architecture_id'])

    new_data[arch_id]["scores"]["grad_norm"]["score"] = '{:f}'.format(bn_info["scores"]["grad_norm"]["score"])
    new_data[arch_id]["scores"]["grad_norm"]["time_usage"] = '{:f}'.format(bn_info["scores"]["grad_norm"]["time_usage"])

    new_data[arch_id]["scores"]["grad_plain"]["score"] = '{:f}'.format(no_bn_info["scores"]["grad_plain"]["score"])
    new_data[arch_id]["scores"]["grad_plain"]["time_usage"] = '{:f}'.format(
        no_bn_info["scores"]["grad_plain"]["time_usage"])

    new_data[arch_id]["scores"]["nas_wot"]["score"] = '{:f}'.format(bn_info["scores"]["nas_wot"]["score"])
    new_data[arch_id]["scores"]["nas_wot"]["time_usage"] = '{:f}'.format(
        bn_info["scores"]["nas_wot"]["time_usage"])

    new_data[arch_id]["scores"]["ntk_cond_num"]["score"] = '{:f}'.format(bn_info["scores"]["ntk_cond_num"]["score"])
    new_data[arch_id]["scores"]["ntk_cond_num"]["time_usage"] = '{:f}'.format(
        bn_info["scores"]["ntk_cond_num"]["time_usage"])

    new_data[arch_id]["scores"]["ntk_trace"]["score"] = '{:f}'.format(no_bn_info["scores"]["ntk_trace"]["score"])
    new_data[arch_id]["scores"]["ntk_trace"]["time_usage"] = '{:f}'.format(
        no_bn_info["scores"]["ntk_trace"]["time_usage"])

    new_data[arch_id]["scores"]["ntk_trace_approx"]["score"] = '{:f}'.format(no_bn_info["scores"]["ntk_trace_approx"]["score"])
    new_data[arch_id]["scores"]["ntk_trace_approx"]["time_usage"] = '{:f}'.format(
        no_bn_info["scores"]["ntk_trace_approx"]["time_usage"])

    new_data[arch_id]["scores"]["fisher"]["score"] = '{:f}'.format(no_bn_info["scores"]["fisher"]["score"])
    new_data[arch_id]["scores"]["fisher"]["time_usage"] = '{:f}'.format(
        no_bn_info["scores"]["fisher"]["time_usage"])

    new_data[arch_id]["scores"]["grasp"]["score"] = '{:f}'.format(bn_info["scores"]["grasp"]["score"])
    new_data[arch_id]["scores"]["grasp"]["time_usage"] = '{:f}'.format(
        bn_info["scores"]["grasp"]["time_usage"])

    new_data[arch_id]["scores"]["snip"]["score"] = '{:f}'.format(bn_info["scores"]["snip"]["score"])
    new_data[arch_id]["scores"]["snip"]["time_usage"] = '{:f}'.format(
        bn_info["scores"]["snip"]["time_usage"])

    new_data[arch_id]["scores"]["synflow"]["score"] = '{:f}'.format(no_bn_info["scores"]["synflow"]["score"])
    new_data[arch_id]["scores"]["synflow"]["time_usage"] = '{:f}'.format(
        no_bn_info["scores"]["synflow"]["time_usage"])


with open("/Users/kevin/project_python/Fast-AutoNAS/main/201_15k_c10_128_union_best_str.json", 'w') as outfile:
    outfile.write(json.dumps(new_data))




