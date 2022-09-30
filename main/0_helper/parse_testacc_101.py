
import itertools
import json
import os
import random

from api import NASBench
from search_space.nas_101_api.lib import nb101_api
from search_space.nas_101_api.lib.nb101_api import ModelSpec
from utilslibs.tools import write_json


def _get_spec(api, arch_hash: str):
    matrix = api.fixed_statistics[arch_hash]['module_adjacency']
    operations = api.fixed_statistics[arch_hash]['module_operations']
    spec = ModelSpec(matrix, operations)
    return spec


def query_api(api, arch_hash, epoch_num):
    res = api.query(_get_spec(api, arch_hash), epochs=epoch_num)
    static = {
        "architecture_id": arch_hash,
        "trainable_parameters": res["trainable_parameters"],
        "time_usage": res["training_time"],
        "train_accuracy": res["train_accuracy"],
        "validation_accuracy": res["validation_accuracy"],
        "test_accuracy": res["test_accuracy"],
    }
    return static


def save_best_score():

    total_num_arch = len(api.hash_iterator())
    arch_id_list = random.sample(range(total_num_arch), total_num_arch)

    parsed_result = {}
    for arch_id in arch_id_list:
        parsed_result[arch_id] = {}
        parsed_result[arch_id]["cifar10"] = {}
        arch_hash = next(itertools.islice(api.hash_iterator(), arch_id, None))
        for epoch in [4, 12, 36, 108]:
            query_info = query_api(api, arch_hash, epoch)
            parsed_result[arch_id]["cifar10"][epoch] = {
                'test-accuracy': query_info["test_accuracy"],
                'time_usage': query_info["time_usage"]
            }

    write_json("101_allEpoch_info", parsed_result)


if __name__ == "__main__":

    base_dir = os.getcwd()
    api_loc = os.path.join(base_dir, "data/nasbench_full.tfrecord")
    api = NASBench(api_loc)
    # api_loc = os.path.join(base_dir, "data/nasbench_only108.pkl")
    # api = nb101_api.NASBench(api_loc)

    save_best_score()

