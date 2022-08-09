from search_space.nas_201_api.lib import NASBench201API
import json

api_loc = "/Users/kevin/project_python/Fast-AutoNAS/data/NAS-Bench-201-v1_1-096897.pth"
api = NASBench201API(api_loc)


# obtain the metric for the `index`-th architecture
# `dataset` indicates the dataset:
#   'cifar10-valid'  : using the proposed train set of CIFAR-10 as the training set
#   'cifar10'        : using the proposed train+valid set of CIFAR-10 as the training set
#   'cifar100'       : using the proposed train set of CIFAR-100 as the training set
#   'ImageNet16-120' : using the proposed train set of ImageNet-16-120 as the training set
# `iepoch` indicates the index of training epochs from 0 to 11/199.
#   When iepoch=None, it will return the metric for the last training epoch
#   When iepoch=11, it will return the metric for the 11-th training epoch (starting from 0)
# `use_12epochs_result` indicates different hyper-parameters for training
#   When use_12epochs_result=True, it trains the network with 12 epochs and the LR decayed from 0.1 to 0 within 12 epochs
#   When use_12epochs_result=False, it trains the network with 200 epochs and the LR decayed from 0.1 to 0 within 200 epochs
# `is_random`
#   When is_random=True, the performance of a random architecture will be returned
#   When is_random=False, the performanceo of all trials will be averaged.
def simulate_train_eval(index, dataset, iepoch, hp, is_random=False):
    info = api.get_more_info(index, dataset, iepoch=iepoch, hp=hp, is_random=is_random)
    test_acc = info["test-accuracy"]
    time_usage = info["train-all-time"] + info["valid-per-time"]
    return test_acc, time_usage


parsed_result = {}
for arch_id in range(0, 15625):
    try:
        parsed_result[arch_id] = {}
        for epoch_num in ["12", "200"]:
            parsed_result[arch_id][epoch_num] = {}
            for dataset in ['cifar10-valid', 'cifar100', 'ImageNet16-120']:
                parsed_result[arch_id][epoch_num][dataset] = {}
                test_acc, time_usage = simulate_train_eval(arch_id, dataset, iepoch=None, hp=epoch_num)
                parsed_result[arch_id][epoch_num][dataset]['test_accuracy'] = test_acc
                parsed_result[arch_id][epoch_num][dataset]['time_usage'] = time_usage
    except:
        pass

with open("201_result_with_time", 'w') as outfile:
    outfile.write(json.dumps(parsed_result))
