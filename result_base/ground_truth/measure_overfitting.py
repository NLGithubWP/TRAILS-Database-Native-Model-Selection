import random

from matplotlib import pyplot as plt

from search_space.nas_201_api.lib import NASBench201API


api_loc = "/Users/kevin/project_python/Fast-AutoNAS/data/NAS-Bench-201-v1_1-096897.pth"
api = NASBench201API(api_loc)

def simulate_train_eval(index, dataset, iepoch, hp, is_random=False):
    info = api.get_more_info(index, dataset, iepoch=iepoch, hp=hp, is_random=is_random)
    test_acc = info["test-accuracy"]
    time_usage = info["train-all-time"] + info["valid-per-time"]
    return test_acc, time_usage

train_acc = []
test_acc = []


for archid in random.sample(range(15624), 400):
    for iepoch in range(200):
        info = api.get_more_info(archid, "cifar10-valid", iepoch=iepoch, hp="200", is_random=False)
        train_acc.append(info["train-accuracy"])
        test_acc.append(info["valid-accuracy"])

    plt.plot(train_acc, label="train-acc")
    plt.plot(test_acc, label="test-acc")
    plt.legend()
    plt.show()
    plt.clf()
