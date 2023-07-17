
import os
import numpy as np

from src.common.constant import Config
from src.utilslibs.io_tools import read_json, write_json

base_dir_folder = os.environ.get("base_dir")
if base_dir_folder is None:base_dir_folder = os.getcwd()
base_dir = os.path.join(base_dir_folder, "result_base/result_append")
print("local api running at {}".format(base_dir))

pre_score_path_101C10 = os.path.join(base_dir, "CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json")

pre_score_path_201C10 = os.path.join(base_dir, "CIFAR10_15625/union/201_15625_c10_bs32_ic16_unionBest.json")
pre_score_path_201C100 = os.path.join(base_dir, "CIFAR100_15625/union/201_15k_c100_bs32_ic16_unionBest.json")
pre_score_path_201IMG = os.path.join(base_dir, "IMAGENET/union/201_15k_imgNet_bs32_ic16_unionBest.json")


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
            topk_index = np.argpartition(score_list[:i+1], -k)[-k:]
            lowerk_index = np.argpartition(score_list[:i+1], k)[:k]

            selected_index = topk_index
            selected_acc = []

            # train top k networks
            for index in selected_index:
                selected_acc.append(acc_list[index])
            result.append( max(selected_acc))
    return result


def convert2pos_(m_name, score):
    if m_name == "grad_norm":
        return score

    if m_name == "grad_plain":
        return -score
    if m_name == "ntk_cond_num":
        return -score
    if m_name == "ntk_trace":
        return score

    if m_name == "ntk_trace_approx":
        return score
    if m_name == "fisher":
        return score
    if m_name == "grasp":
        return score
    if m_name == "snip":
        return score
    if m_name == "synflow":
        return score
    if m_name == "nas_wot":
        return score
    if m_name == "jacob_conv":
        return score
    if m_name == "weight_norm":
        return score

    return score


class LocalApi:

    def __init__(self, search_space_name, dataset):
        self.data = None
        if search_space_name == Config.NB201:
            if dataset == Config.c10:
                self.pre_score_path = pre_score_path_201C10
            elif dataset == Config.c100:
                self.pre_score_path = pre_score_path_201C100
            elif dataset == Config.imgNet:
                self.pre_score_path = pre_score_path_201IMG

        if search_space_name == Config.NB101:
            self.pre_score_path = pre_score_path_101C10

    def api_get_score(self, arch_id, algName_m):
        # retrieve score from pre-scored file
        self.lazy_load_data()
        ori_score = float(self.data[arch_id][algName_m])
        return convert2pos_(algName_m, ori_score)

    def update_existing_data(self, arch_id, alg_name, score_str):
        """
        Add new arch's inf into data
        :param arch_id:
        :param alg_name:
        :param score_str:
        :return:
        """
        self.lazy_load_data()
        if str(arch_id) not in self.data:
            self.data[str(arch_id)] = {}
        else:
            self.data[str(arch_id)] = self.data[str(arch_id)]
        self.data[str(arch_id)][alg_name] = '{:f}'.format(score_str)

    def is_arch_and_alg_inside_data(self, arch_id, alg_name):
        self.lazy_load_data()
        if arch_id in self.data and alg_name in self.data[arch_id]:
            return True
        else:
            return False

    def is_arch_inside_data(self, arch_id):
        self.lazy_load_data()
        if arch_id in self.data:
            return True
        else:
            return False

    def get_len_data(self):
        self.lazy_load_data()
        return len(self.data)

    def save_latest_data(self):
        """
        update the latest score data
        """
        write_json(self.pre_score_path, self.data)

    def lazy_load_data(self):
        """
        Read the pre-score-data
        :return:
        """
        if self.data is None:
            self.data = read_json(self.pre_score_path)
            print("localApi init, len(data) = ", len(list(self.data.keys())))


if __name__ == "__main__":
    lapi = LocalApi(Config.NB101, Config.c10)
    lapi.get_len_data()
