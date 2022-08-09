import os

from utilslibs.tools import read_json

base_dir = os.getcwd()
print("gt_api running at {}".format(base_dir))
gt201 = os.path.join(base_dir, "result_base/ground_truth/201_result_with_time")
gt101 = os.path.join(base_dir, "result_base/ground_truth/101_result_with_time")

c10_valid = "cifar10-valid"
c10 = "cifar10"
c100 = "cifar100"
imgNet = "ImageNet16-120"


class Gt201:
    data201 = read_json(gt201)

    @classmethod
    def get_c10_200epoch_test_info(cls, arch_id: int):
        """
        cifar10-valid means train with train set, valid with validation dataset
        Thus, acc is lower than train with train+valid.
        :param arch_id:
        :return:
        """
        return cls._query_200_epoch(str(arch_id), c10_valid)

    @classmethod
    def get_c100_200epoch_test_info(cls, arch_id: int):
        return cls._query_200_epoch(str(arch_id), c100)

    @classmethod
    def get_imgNet_200epoch_test_info(cls, arch_id: int):
        return cls._query_200_epoch(str(arch_id), imgNet)

    @staticmethod
    def _query_200_epoch(arch_id: str, dataset):
        t_acc = Gt201.data201[arch_id]["200"][dataset]["test_accuracy"]
        time_usage = Gt201.data201[arch_id]["200"][dataset]["time_usage"]
        return t_acc, time_usage

    @classmethod
    def count_models(cls):
        return len(cls.data201)


class Gt101:
    data101 = read_json(gt101)

    @classmethod
    def get_c10_test_info(cls, arch_id: int):
        """
        cifar10-valid means train with train set, valid with validation dataset
        Thus, acc is lower than train with train+valid.
        :param arch_id_int:
        :return:
        """

        t_acc = cls.data101[str(arch_id)][c10]["test-accuracy"]
        time_usage = cls.data101[str(arch_id)][c10]["time_usage"]
        return t_acc, time_usage

    @classmethod
    def count_models(cls):
        return len(cls.data101)



