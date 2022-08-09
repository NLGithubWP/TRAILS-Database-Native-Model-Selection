import json
import os
import numpy as np
import scipy.stats as ss


class LocalApi:

    def __init__(self, base_path, gt_path, space, space_name):
        self.gt = None
        self.base_path = base_path
        self.gt_path = gt_path
        self.space = space
        self.space_name = space_name
        self.data = None

    # by default, greater the better
    def convert2pos(self, m_name, score):
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

    def api_get_score(self, arch_id, algName_m):
        self.lazy_load_data()
        ori_score = float(self.data[arch_id][algName_m])
        return self.convert2pos(algName_m, ori_score)

    def api_get_ground_truth(self, arch_id, dataset):
        if "101" in self.space_name:
            # begin = time.time()
            try:
                result = float(self.space.query_performance(int(arch_id), dataset)["test_accuracy"])
                # print(time.time() - begin)
                return result
            except:
                print("arch_id = ", arch_id)

        elif "201" in self.space_name:
            self.lazy_load_gt_file()
            return 0.01 * float(self.gt[arch_id][dataset]["test-accuracy"])
        else:
            raise

    def api_get_acc(self, acc_list):
        """
        Get latest greatest accuracy
        :param acc_list:
        :return:
        """
        result = [acc_list[0]]
        for i in range(1, len(acc_list), 1):
            if acc_list[i] > result[-1]:
                result.append(acc_list[i])
            else:
                result.append(result[-1])

        return result

    def get_rank_score(self, score_list):
        """
        Get rank / len of current
        :param score_list:
        :return:
        """
        rank_index_list = ss.rankdata(score_list)
        return rank_index_list[-1] / len(score_list)

    def api_simulate_evaluate(self, acc_list, score_list, k):
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
                    selected_acc.append(self.train_networks(acc_list, index))
                result.append( max(selected_acc))
        return result

    def train_networks(self, acc_list, index):
        """
        Return accuracy of the give index arch
        :param acc_list:
        :param index:
        :return:
        """
        return acc_list[index]

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

    def save_latest_data(self):
        """
        Save latest score
        :return:
        """
        with open(self.base_path, 'w') as outfile:
            outfile.write(json.dumps(self.data))

    def lazy_load_data(self):
        if self.data is None:
            with open(self.base_path, 'r') as readfile:
                self.data = json.load(readfile)
            print("localApi init, len(data) = ", len(list(self.data.keys())), "len(gt) =")

    def lazy_load_gt_file(self):
        if os.path.exists(self.gt_path):
            if self.gt is None:
                with open(self.gt_path, 'r') as readfile:
                    self.gt = json.load(readfile)

    def is_arch_inside_data(self, arch_id, alg_name):
        self.lazy_load_data()
        if arch_id in self.data and alg_name in self.data[arch_id]:
            return True
        else:
            return False


if __name__ == "__main__":

    loapi = LocalApi("./result_append/CIFAR10_15625/union/101_15k_c10_bs32_ic16_unionBest.json")



