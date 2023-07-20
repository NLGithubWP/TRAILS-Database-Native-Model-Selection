import os
import numpy as np

from src.common.constant import Config
from src.utilslibs.io_tools import read_json, write_json

base_dir_folder = os.environ.get("base_dir")
if base_dir_folder is None: base_dir_folder = os.getcwd()
base_dir = os.path.join(base_dir_folder, "img_data")
print("local api running at {}".format(base_dir))

pre_score_path_101C10 = os.path.join(base_dir, "score_101_15k_c10_128.json")
pre_score_path_201C10 = os.path.join(base_dir, "score_201_15k_c10_bs32_ic16.json")
pre_score_path_201C100 = os.path.join(base_dir, "score_201_15k_c100_bs32_ic16.json")
pre_score_path_201IMG = os.path.join(base_dir, "score_201_15k_imgNet_bs32_ic16.json")


def api_get_current_best_acc(acc_list):
    """
    Get latest greatest accuracy
    :param acc_list: a list of accuracy
    :return:
    """
    result = [acc_list[0]]
    for i in range(1, len(acc_list), 1):
        if acc_list[i] > result[-1]:
            result.append(acc_list[i])
        else:
            result.append(result[-1])

    return result


def api_simulate_evaluate(acc_list, score_list, k):
    """
    Get best acc among the top_k-score archs
    :param acc_list:
    :param score_list:
    :param k:
    :return:
    """
    result = []
    for i in range(len(score_list)):
        # if i < k, train_evaluate them
        if i < k:
            result.append(acc_list[i])
        else:
            # get top or lower
            topk_index = np.argpartition(score_list[:i + 1], -k)[-k:]
            lowerk_index = np.argpartition(score_list[:i + 1], k)[:k]

            selected_index = topk_index
            selected_acc = []

            # train top k networks
            for index in selected_index:
                selected_acc.append(acc_list[index])
            result.append(max(selected_acc))
    return result


class LocalApi:
    # Multiton pattern
    _instances = {}

    def __new__(cls, search_space_name: str, dataset: str):
        if (search_space_name, dataset) not in cls._instances:
            instance = super(LocalApi, cls).__new__(cls)
            instance.search_space_name, instance.dataset = search_space_name, dataset

            # read pre-scored file path
            if search_space_name == Config.NB201:
                if dataset == Config.c10:
                    instance.pre_score_path = pre_score_path_201C10
                elif dataset == Config.c100:
                    instance.pre_score_path = pre_score_path_201C100
                elif dataset == Config.imgNet:
                    instance.pre_score_path = pre_score_path_201IMG
            if search_space_name == Config.NB101:
                instance.pre_score_path = pre_score_path_101C10

            instance.data = read_json(instance.pre_score_path)
            cls._instances[(search_space_name, dataset)] = instance
        return cls._instances[(search_space_name, dataset)]

    def api_get_score(self, arch_id: str, tfmem: str = None):
        # retrieve score from pre-scored file
        if tfmem is None:
            return self.data[arch_id]
        else:
            return {tfmem: float(self.data[arch_id][tfmem])}

    def update_existing_data(self, arch_id, alg_name, score_str):
        """
        Add new arch's inf into data
        :param arch_id:
        :param alg_name:
        :param score_str:
        :return:
        """
        if str(arch_id) not in self.data:
            self.data[str(arch_id)] = {}
        else:
            self.data[str(arch_id)] = self.data[str(arch_id)]
        self.data[str(arch_id)][alg_name] = '{:f}'.format(score_str)

    def is_arch_and_alg_inside_data(self, arch_id, alg_name):
        if arch_id in self.data and alg_name in self.data[arch_id]:
            return True
        else:
            return False

    def is_arch_inside_data(self, arch_id):
        if arch_id in self.data:
            return True
        else:
            return False

    def get_len_data(self):
        return len(self.data)

    def save_latest_data(self):
        """
        update the latest score data
        """
        write_json(self.pre_score_path, self.data)

    def get_all_scored_model_ids(self):
        return list(self.data.keys())


if __name__ == "__main__":
    lapi = LocalApi(Config.NB101, Config.c10)
    lapi.get_len_data()
