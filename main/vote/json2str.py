

import json

with open("/Users/kevin/project_python/Fast-AutoNAS/Logs/cifar100_15000/201_15k_c100_128_noBN.json", 'r') as readfile:
    data = json.load(readfile)


print("data is loaded")

new_data = {}

required = len(data.keys())
num = 0
for ele in data.keys():
    if num == required:
        break
    new_data[ele] = data[ele]
    new_data[ele]['trainable_parameters'] = ""
    new_data[ele]['training_time'] = ""
    new_data[ele]['train_accuracy'] = str(new_data[ele]['train_accuracy'])
    new_data[ele]['validation_accuracy'] = str(new_data[ele]['validation_accuracy'])
    new_data[ele]['test_accuracy'] = str(new_data[ele]['test_accuracy'])
    new_data[ele]['architecture_id'] = str(new_data[ele]['architecture_id'])

    for alg_name, info in new_data[ele]["scores"].items():
        new_data[ele]["scores"][alg_name]["score"] = '{:f}'.format(new_data[ele]["scores"][alg_name]["score"])
        new_data[ele]["scores"][alg_name]["time_usage"] = '{:f}'.format(new_data[ele]["scores"][alg_name]["time_usage"])

    num += 1


with open("./Logs/cifar100_15000/201_15k_c100_128_noBN.json", 'w') as outfile:
    outfile.write(json.dumps(new_data))




