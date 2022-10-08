import os
import random
import time

from common.constant import Config
from utilslibs.tools import read_json

base_dir = os.getcwd()
print("gt_api running at {}".format(base_dir))
gt201 = os.path.join(base_dir, "result_base/ground_truth/201_allEpoch_info")
gt101 = os.path.join(base_dir, "result_base/ground_truth/101_allEpoch_info")


class Gt201:
    data201 = read_json(gt201)

    @classmethod
    def get_c10valid_200epoch_test_info(cls, arch_id: int):
        """
        cifar10-valid means train with train set, valid with validation dataset
        Thus, acc is lower than train with train+valid.
        :param arch_id:
        :return:
        """
        return cls.query_200_epoch(str(arch_id), Config.c10_valid)

    @classmethod
    def get_c10_200epoch_test_info(cls, arch_id: int):
        """
        cifar10-valid means train with train set, valid with validation dataset
        Thus, acc is lower than train with train+valid.
        :param arch_id:
        :return:
        """
        return cls.query_200_epoch(str(arch_id), Config.c10)

    @classmethod
    def get_c100_200epoch_test_info(cls, arch_id: int):
        return cls.query_200_epoch(str(arch_id), Config.c100)

    @classmethod
    def get_imgNet_200epoch_test_info(cls, arch_id: int):
        return cls.query_200_epoch(str(arch_id), Config.imgNet)

    @staticmethod
    def query_200_epoch(arch_id: str, dataset, epoch_num: int = 199):
        if epoch_num > 199:
            epoch_num = 199
        arch_id = str(arch_id)
        t_acc = Gt201.data201[arch_id]["200"][dataset][str(epoch_num)]["test_accuracy"] * 0.01
        time_usage = Gt201.data201[arch_id]["200"][dataset][str(epoch_num)]["time_usage"]
        return t_acc, time_usage

    @classmethod
    def count_models(cls):
        return len(cls.data201)

    @classmethod
    def guess_eval_time(cls):
        return random.randint(3315, 4502) * 0.0001


class Gt101:
    data101 = read_json(gt101)

    @classmethod
    def get_c10_test_info(cls, arch_id: int, epoch_num: int = 108):
        """
        Default use 108 epoch for c10, this is the largest epoch number.
        :param arch_id: architecture id
        :param epoch_num: query the result of the specific epoch number
        :return:
        """
        if epoch_num > 108:
            epoch_num = 108
        elif epoch_num > 36:
            epoch_num = 36
        elif epoch_num > 12:
            epoch_num = 12
        elif epoch_num > 4:
            epoch_num = 4
        else:
            epoch_num = 4

        t_acc = cls.data101[str(arch_id)][Config.c10][str(epoch_num)]["test-accuracy"]
        time_usage = cls.data101[str(arch_id)][Config.c10][str(epoch_num)]["time_usage"]
        return t_acc, time_usage

    @classmethod
    def count_models(cls):
        return len(cls.data101)

    @classmethod
    def guess_eval_time(cls):
        return random.randint(1169, 1372) * 0.0001


if __name__ == "__main__":
    gt201 = Gt201()
    begin_time201 = time.time()
    test_accuracy, time_usage = gt201.query_200_epoch(arch_id=str(123), dataset=Config.imgNet, epoch_num=1)
    end_time = time.time()
    print(test_accuracy, time_usage, end_time-begin_time201)

    # 101 time measurement
    begin_time101 = time.time()
    gt101 = Gt101()
    test_accuracy, time_usage = gt101.get_c10_test_info(arch_id=123)
    end_time = time.time()
    print(test_accuracy, time_usage, end_time - begin_time101)


