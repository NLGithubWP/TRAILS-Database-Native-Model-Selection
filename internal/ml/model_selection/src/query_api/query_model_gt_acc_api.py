import os
import random
import threading
import time

from src.common.constant import Config
from src.utilslibs.compute import binary_insert_get_rank, load_global_rank
from src.utilslibs.io_tools import read_json, read_pickle

base_dir = os.environ.get("base_dir")
if base_dir is None: base_dir = os.getcwd()
print("base_dir is {}".format(base_dir))
gt201 = os.path.join(base_dir, "result_base/ground_truth/201_allEpoch_info")

gt101 = os.path.join(base_dir, "result_base/ground_truth/101_allEpoch_info_json")
gt101P = os.path.join(base_dir, "result_base/ground_truth/nasbench1_accuracy.p")
id_to_hash_path = os.path.join(base_dir, "result_base/ground_truth/nb101_id_to_hash.json")

# MLP related ground truth
mlp_train_frappe = os.path.join(base_dir, "result_base/mlp_results/frappe/all_train_baseline_frappe.json")
mlp_train_uci_diabetes = os.path.join(base_dir, "result_base/mlp_results/uci_diabetes/"
                                                "all_train_baseline_uci_160k_40epoch.json")
mlp_train_criteo = os.path.join(base_dir, "result_base/mlp_results/criteo/all_train_baseline_criteo.json")

mlp_score_frappe = os.path.join(base_dir, "result_base/mlp_results/frappe/"
                                          "score_frappe_batch_size_32_nawot_synflow.json")
mlp_score_uci_diabetes = os.path.join(base_dir, "result_base/mlp_results/uci_diabetes/"
                                                "score_uci_diabetes_batch_size_32_all_metrics.json")
mlp_score_criteo = os.path.join(base_dir, "result_base/mlp_results/criteo/score_criteo_batch_size_32.json")


def guess_score_time(search_space_m, dataset):
    if search_space_m == Config.NB101:
        return Gt101.guess_score_time()
    if search_space_m == Config.NB201:
        return Gt201.guess_score_time(dataset)


def guess_train_one_epoch_time(search_space_m, dataset):
    if search_space_m == Config.NB101:
        return Gt101().guess_train_one_epoch_time()
    if search_space_m == Config.NB201:
        return Gt201().guess_train_one_epoch_time(dataset)


def profile_NK_trade_off(dataset):
    if dataset == Config.c10:
        return 85
    elif dataset == Config.c100:
        return 85
    elif dataset == Config.imgNet:
        return 130
    else:
        return 100


class Singleton(object):
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(Singleton, "_instance"):
            with Singleton._instance_lock:
                if not hasattr(Singleton, "_instance"):
                    Singleton._instance = object.__new__(cls)
        return Singleton._instance


class Gt201(Singleton):
    # multiple instance share the class variables.
    _instance_lock = threading.Lock()
    data201 = None

    def load_201(self):
        if self.data201 is None:
            self.data201 = read_json(gt201)

    def get_c10valid_200epoch_test_info(self, arch_id: int):
        """
        cifar10-valid means train with train set, valid with validation dataset
        Thus, acc is lower than train with train+valid.
        :param arch_id:
        :return:
        """
        return self.query_200_epoch(str(arch_id), Config.c10_valid)

    def get_c10_200epoch_test_info(self, arch_id: int):
        """
        cifar10-valid means train with train set, valid with validation dataset
        Thus, acc is lower than train with train+valid.
        :param arch_id:
        :return:
        """
        return self.query_200_epoch(str(arch_id), Config.c10)

    def get_c100_200epoch_test_info(self, arch_id: int):
        return self.query_200_epoch(str(arch_id), Config.c100)

    def get_imgNet_200epoch_test_info(self, arch_id: int):
        return self.query_200_epoch(str(arch_id), Config.imgNet)

    def query_200_epoch(self, arch_id: str, dataset, epoch_num: int = 199):
        self.load_201()
        if epoch_num is None or epoch_num > 199:
            epoch_num = 199
        arch_id = str(arch_id)
        t_acc = self.data201[arch_id]["200"][dataset][str(epoch_num)]["test_accuracy"]
        time_usage = self.data201[arch_id]["200"][dataset][str(epoch_num)]["time_usage"]
        return t_acc, time_usage

    def query_12_epoch(self, arch_id: str, dataset, epoch_num: int = 11):
        self.load_201()
        if epoch_num is None or epoch_num > 11:
            epoch_num = 11
        arch_id = str(arch_id)
        t_acc = self.data201[arch_id]["12"][dataset][str(epoch_num)]["test_accuracy"]
        time_usage = self.data201[arch_id]["12"][dataset][str(epoch_num)]["time_usage"]
        return t_acc, time_usage

    def count_models(self):
        return len(self.data201)

    @classmethod
    def guess_score_time(cls, dataset=Config.c10):
        return random.randint(3315, 4502) * 0.0001

    def guess_train_one_epoch_time(self, dataset):
        if dataset == Config.c10:
            dataset = Config.c10_valid
        # pick the max value over 5k arch training time, it's 40
        # res = 0
        # for arch_id in range(15624):
        #     _, time_usage = self.query_200_epoch(str(arch_id), dataset, 1)
        #     if time_usage > res:
        #         res = time_usage
        # return res
        return 40


class Gt101(Singleton):
    # multiple instance share the class variables.
    data101_from_zerocost = None
    id_to_hash_map = None
    data101_full = None

    def load_101(self):
        if self.data101_from_zerocost is None:
            self.data101_from_zerocost = read_pickle(gt101P)
            self.id_to_hash_map = read_json(id_to_hash_path)
            self.data101_full = read_json(gt101)

    def get_c10_test_info(self, arch_id: str, dataset: str = Config.c10, epoch_num: int = 108):
        """
        Default use 108 epoch for c10, this is the largest epoch number.
        :param dataset:
        :param arch_id: architecture id
        :param epoch_num: query the result of the specific epoch number
        :return:
        """
        self.load_101()
        if dataset != Config.c10:
            raise "NB101 only have c10 results"

        if epoch_num is None or epoch_num > 108:
            epoch_num = 108
        elif epoch_num > 36:
            epoch_num = 36
        elif epoch_num > 12:
            epoch_num = 12
        elif epoch_num > 4:
            epoch_num = 4
        else:
            epoch_num = 4
        arch_id = str(arch_id)
        # this is acc from zero-cost paper, which only record 108 epoch' result [test, valid, train]
        # t_acc = self.data101_from_zerocost[self.id_to_hash_map[arch_id]][0]
        # this is acc from parse_testacc_101.py,
        t_acc_usage = self.data101_full[arch_id][Config.c10][str(epoch_num)]["test-accuracy"]
        time_usage = self.data101_full[arch_id][Config.c10][str(epoch_num)]["time_usage"]
        # print(f"[Debug]: Acc different = {t_acc_usage - t_acc}")
        return t_acc_usage, time_usage

    def count_models(self):
        return len(self.data101_from_zerocost)

    @classmethod
    def guess_score_time(cls):
        return random.randint(1169, 1372) * 0.0001

    def guess_train_one_epoch_time(self):
        # only have information for 4 epoch
        self.load_101()

        d = dict.fromkeys(self.data101_full)
        keys = random.sample(list(d), 15000)

        # pick the max value over 5k arch training time
        res = 0
        for rep_time in range(15000):
            arch_id = keys[rep_time]
            _, time_usage = self.get_c10_test_info(arch_id=arch_id, dataset=Config.c10, epoch_num=4)
            if time_usage > res:
                res = time_usage
        return res


class GTMLP(Singleton):
    # multiple instance share the class variables.
    _instance_lock = threading.Lock()

    default_alg_name_list = ["nas_wot", "synflow"]
    device = "cpu"

    mlp_frappe_train = None
    mlp_frappe_score = None
    mlp_frappe_global_rank = {}

    mlp_criteo_train = None
    mlp_criteo_score = None
    mlp_criteo_global_rank = {}

    mlp_uci_train = None
    mlp_uci_score = None
    mlp_uci_global_rank = {}

    @staticmethod
    def get_score_one_model_time(dataset: str, device: str):
        # those are got from offline training, CPU only, GPU only, second
        if device == "cpu":
            if dataset == Config.Frappe:
                _train_time_per_epoch = 0.0211558125
            elif dataset == Config.UCIDataset:
                _train_time_per_epoch = 0.015039052631578948
            elif dataset == Config.Criteo:
                _train_time_per_epoch = 0.6824370454545454
            else:
                raise NotImplementedError
        else:
            if dataset == Config.Frappe:
                _train_time_per_epoch = 0.013744457142857143
            elif dataset == Config.UCIDataset:
                _train_time_per_epoch = 0.008209692307692308
            elif dataset == Config.Criteo:
                _train_time_per_epoch = 0.6095493157894737
            else:
                raise NotImplementedError
        return _train_time_per_epoch

    @staticmethod
    def get_train_one_epoch_time(dataset: str, device: str):
        # those are got from offline training, CPU only, GPU only
        if device == "cpu":
            if dataset == Config.Frappe:
                _train_time_per_epoch = 5.122203075885773
            elif dataset == Config.UCIDataset:
                _train_time_per_epoch = 4.16297769
            elif dataset == Config.Criteo:
                _train_time_per_epoch = 422
            else:
                raise NotImplementedError
        else:
            if dataset == Config.Frappe:
                _train_time_per_epoch = 2.8
            elif dataset == Config.UCIDataset:
                _train_time_per_epoch = 1.4
            elif dataset == Config.Criteo:
                _train_time_per_epoch = 125
            else:
                raise NotImplementedError
        return _train_time_per_epoch

    def load_mlp_train(self, dataset):
        if dataset == Config.Frappe:
            if self.mlp_frappe_train is None:
                self.mlp_frappe_train = read_json(mlp_train_frappe)

        if dataset == Config.Criteo:
            if self.mlp_criteo_train is None:
                self.mlp_criteo_train = read_json(mlp_train_criteo)

        if dataset == Config.UCIDataset:
            if self.mlp_uci_train is None:
                self.mlp_uci_train = read_json(mlp_train_uci_diabetes)

    def load_mlp_score(self, dataset):
        if dataset == Config.Frappe:
            if self.mlp_frappe_score is None:
                self.mlp_frappe_score = read_json(mlp_score_frappe)

        if dataset == Config.Criteo:
            if self.mlp_criteo_score is None:
                self.mlp_criteo_score = read_json(mlp_score_criteo)

        if dataset == Config.UCIDataset:
            if self.mlp_uci_score is None:
                self.mlp_uci_score = read_json(mlp_score_uci_diabetes)

    def load_mlp_global_score_rank(self, dataset):

        if dataset == Config.Frappe:
            self.load_mlp_score(dataset)
            if self.mlp_frappe_global_rank == {}:
                self.mlp_frappe_global_rank = load_global_rank(self.mlp_frappe_score, self.default_alg_name_list)

        if dataset == Config.Criteo:
            self.load_mlp_score(dataset)
            if self.mlp_criteo_global_rank == {}:
                self.mlp_criteo_global_rank = load_global_rank(self.mlp_criteo_score, self.default_alg_name_list)

        if dataset == Config.UCIDataset:
            self.load_mlp_score(dataset)
            if self.mlp_uci_global_rank == {}:
                self.mlp_uci_global_rank = load_global_rank(self.mlp_uci_score, self.default_alg_name_list)

    def get_valid_auc(self, arch_id: str, dataset, epoch_num: int):
        self.load_mlp_train(dataset)
        # todo: due to the too many job contention on server, the time usage may not valid.
        time_usage = (int(epoch_num) + 1) * self.get_train_one_epoch_time(dataset, self.device)
        if dataset == Config.Frappe:
            if epoch_num is None or epoch_num >= 20: epoch_num = 19
            t_acc = self.mlp_frappe_train[dataset][arch_id][str(epoch_num)]["valid_auc"]
            # time_usage = self.mlp_frappe_train[dataset][arch_id][str(epoch_num)]["train_val_total_time"]
            return t_acc, time_usage
        elif dataset == Config.Criteo:
            if epoch_num is None or epoch_num >= 10: epoch_num = 9
            t_acc = self.mlp_criteo_train[dataset][arch_id][str(epoch_num)]["valid_auc"]
            # time_usage = self.mlp_criteo_train[dataset][arch_id][str(epoch_num)]["train_val_total_time"]
            return t_acc, time_usage
        elif dataset == Config.UCIDataset:
            if epoch_num is None or epoch_num >= 40: epoch_num = 39
            t_acc = self.mlp_uci_train[dataset][arch_id][str(epoch_num)]["valid_auc"]
            # time_usage = self.mlp_uci_train[dataset][arch_id][str(epoch_num)]["train_val_total_time"]
            return t_acc, time_usage
        elif dataset == Config.Criteo:
            if epoch_num is None: epoch_num = 9
            t_acc = self.mlp_criteo_train[dataset][arch_id][str(epoch_num)]["valid_auc"]
            time_usage = self.mlp_criteo_train[dataset][arch_id][str(epoch_num)]["train_val_total_time"]
            return t_acc, time_usage
        elif dataset == Config.UCIDataset:
            if epoch_num is None: epoch_num = 39
            t_acc = self.mlp_uci_train[dataset][arch_id][str(epoch_num)]["valid_auc"]
            time_usage = self.mlp_uci_train[dataset][arch_id][str(epoch_num)]["train_val_total_time"]
            return t_acc, time_usage
        else:
            raise NotImplementedError

    def get_metrics_score(self, arch_id: str, dataset) -> dict:
        self.load_mlp_score(dataset)
        if dataset == Config.Frappe:
            score_dic = self.mlp_frappe_score[arch_id]
            return score_dic
        elif dataset == Config.Criteo:
            score_dic = self.mlp_criteo_score[arch_id]
            return score_dic
        elif dataset == Config.UCIDataset:
            score_dic = self.mlp_uci_score[arch_id]
            return score_dic
        else:
            raise NotImplementedError

    def get_global_rank_score(self, arch_id, dataset):
        self.load_mlp_global_score_rank(dataset)
        if dataset == Config.Frappe:
            return self.mlp_frappe_global_rank[arch_id]
        elif dataset == Config.Criteo:
            return self.mlp_criteo_global_rank[arch_id]
        elif dataset == Config.UCIDataset:
            return self.mlp_uci_global_rank[arch_id]
        else:
            raise NotImplementedError


if __name__ == "__main__":

    # 101 time measurement
    begin_time101 = time.time()
    gt101_ins = Gt101()
    test_accuracy, time_usage = gt101_ins.get_c10_test_info(arch_id=str(123))
    end_time = time.time()
    print(test_accuracy, time_usage, end_time - begin_time101)

    # 201 time measurement
    gt201_ins = Gt201()
    begin_time201 = time.time()
    test_accuracy, time_usage = gt201_ins.query_12_epoch(arch_id=str(35), dataset=Config.c10_valid)
    end_time = time.time()
    print(test_accuracy, time_usage, end_time - begin_time201)

    res = []
    for arch in range(1, 15615):
        res.append(gt201_ins.query_200_epoch(arch_id=str(arch), dataset=Config.c10_valid)[0])

    print("Max accuracy in NB201 with c10_valid is ", max(res))
