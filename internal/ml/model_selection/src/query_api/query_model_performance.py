import os
import random
import threading
from src.common.constant import Config
from src.utilslibs.compute import load_global_rank
from src.utilslibs.io_tools import read_json, read_pickle

base_dir = os.environ.get("base_dir")
if base_dir is None: base_dir = os.getcwd()
print("base_dir is {}".format(base_dir))
gt201 = os.path.join(base_dir, "img_data/ground_truth/201_allEpoch_info")

gt101 = os.path.join(base_dir, "img_data/ground_truth/101_allEpoch_info_json")
gt101P = os.path.join(base_dir, "img_data/ground_truth/nasbench1_accuracy.p")
id_to_hash_path = os.path.join(base_dir, "img_data/ground_truth/nb101_id_to_hash.json")

# MLP related ground truth
mlp_train_frappe = os.path.join(base_dir, "tab_data/frappe/all_train_baseline_frappe.json")

mlp_train_uci_diabetes = os.path.join(base_dir,
                                      "tab_data/uci_diabetes/all_train_baseline_uci_160k_40epoch.json")

mlp_train_criteo = os.path.join(base_dir, "tab_data/criteo/all_train_baseline_criteo.json")

# score result
mlp_score_frappe = os.path.join(base_dir, "tab_data/frappe/score_frappe_batch_size_32_local_finish_all_models.json")
# mlp_score_frappe = os.path.join(base_dir, "tab_data/frappe/score_frappe_batch_size_32_nawot_synflow.json")
mlp_score_uci = os.path.join(base_dir, "tab_data/uci_diabetes/score_uci_diabetes_batch_size_32_all_metrics.json")
mlp_score_criteo = os.path.join(base_dir, "tab_data/criteo/score_criteo_batch_size_32.json")

# pre computed result
score_one_model_time_dict = {
    "cpu": {
        Config.Frappe: 0.0211558125,
        Config.UCIDataset: 0.015039052631578948,
        Config.Criteo: 0.6824370454545454
    },
    "gpu": {
        Config.Frappe: 0.013744457142857143,
        Config.UCIDataset: 0.008209692307692308,
        Config.Criteo: 0.6095493157894737
    }
}

train_one_epoch_time_dict = {
    "cpu": {
        Config.Frappe: 5.122203075885773,
        Config.UCIDataset: 4.16297769,
        Config.Criteo: 422
    },
    "gpu": {
        Config.Frappe: 2.8,
        Config.UCIDataset: 1.4,
        Config.Criteo: 125
    }
}


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


class Singleton(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Gt201(metaclass=Singleton):

    @classmethod
    def guess_score_time(cls, dataset=Config.c10):
        return random.randint(3315, 4502) * 0.0001

    def __init__(self):
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
        if epoch_num is None or epoch_num > 199:
            epoch_num = 199
        arch_id = str(arch_id)
        t_acc = self.data201[arch_id]["200"][dataset][str(epoch_num)]["test_accuracy"]
        time_usage = self.data201[arch_id]["200"][dataset][str(epoch_num)]["time_usage"]
        return t_acc, time_usage

    def query_12_epoch(self, arch_id: str, dataset, epoch_num: int = 11):
        if epoch_num is None or epoch_num > 11:
            epoch_num = 11
        arch_id = str(arch_id)
        t_acc = self.data201[arch_id]["12"][dataset][str(epoch_num)]["test_accuracy"]
        time_usage = self.data201[arch_id]["12"][dataset][str(epoch_num)]["time_usage"]
        return t_acc, time_usage

    def count_models(self):
        return len(self.data201)

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

    def get_all_trained_model_ids(self):
        # 201 all data has the same model set.
        return list(self.data201.keys())


class Gt101(metaclass=Singleton):

    @classmethod
    def guess_score_time(cls):
        return random.randint(1169, 1372) * 0.0001

    def __init__(self):
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

    def guess_train_one_epoch_time(self):
        # only have information for 4 epoch
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

    def get_all_trained_model_ids(self):
        return list(self.data101_full.keys())


class GTMLP:
    _instances = {}
    default_alg_name_list = ["nas_wot", "synflow"]
    device = "cpu"

    def __new__(cls, dataset: str):
        if dataset not in cls._instances:
            instance = super(GTMLP, cls).__new__(cls)
            instance.dataset = dataset
            if dataset == Config.Frappe:
                instance.mlp_train_path = mlp_train_frappe
                instance.mlp_score_path = mlp_score_frappe
            elif dataset == Config.Criteo:
                instance.mlp_train_path = mlp_train_criteo
                instance.mlp_score_path = mlp_score_criteo
            elif dataset == Config.UCIDataset:
                instance.mlp_train_path = mlp_train_uci_diabetes
                instance.mlp_score_path = mlp_score_uci
            instance.mlp_train = read_json(instance.mlp_train_path)
            instance.mlp_score = read_json(instance.mlp_score_path)
            instance.mlp_global_rank = load_global_rank(
                instance.mlp_score, instance.default_alg_name_list)

            cls._instances[dataset] = instance
        return cls._instances[dataset]

    def get_all_trained_model_ids(self):
        return list(self.mlp_train[self.dataset].keys())

    def get_all_scored_model_ids(self):
        return list(self.mlp_score.keys())

    def get_score_one_model_time(self, device: str):
        _train_time_per_epoch = score_one_model_time_dict[device].get(self.dataset)
        if _train_time_per_epoch is None:
            raise NotImplementedError
        return _train_time_per_epoch

    def get_train_one_epoch_time(self, device: str):
        _train_time_per_epoch = train_one_epoch_time_dict[device].get(self.dataset)
        if _train_time_per_epoch is None:
            raise NotImplementedError
        return _train_time_per_epoch

    def get_valid_auc(self, arch_id: str, epoch_num: int):
        # todo: due to the too many job contention on server, the time usage may not valid.
        time_usage = (int(epoch_num) + 1) * self.get_train_one_epoch_time(self.device)
        if self.dataset == Config.Frappe:
            if epoch_num is None or epoch_num >= 20: epoch_num = 19
            t_acc = self.mlp_train[self.dataset][arch_id][str(epoch_num)]["valid_auc"]
            return t_acc, time_usage
        elif self.dataset == Config.Criteo:
            if epoch_num is None or epoch_num >= 10: epoch_num = 9
            t_acc = self.mlp_train[self.dataset][arch_id][str(epoch_num)]["valid_auc"]
            return t_acc, time_usage
        elif self.dataset == Config.UCIDataset:
            if epoch_num is None or epoch_num >= 40: epoch_num = 39
            t_acc = self.mlp_train[self.dataset][arch_id][str(epoch_num)]["valid_auc"]
            return t_acc, time_usage
        else:
            raise NotImplementedError

    def api_get_score(self, arch_id: str) -> dict:
        score_dic = self.mlp_score[arch_id]
        return score_dic

    def get_global_rank_score(self, arch_id):
        return self.mlp_global_rank[arch_id]
