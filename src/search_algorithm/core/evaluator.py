import time
from abc import abstractmethod
import torch

from logger import logger
from search_algorithm.utils.gpu_util import showUtilization


class Evaluator:
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, arch, pre_defined, batch_data: torch.tensor, batch_labels: torch.tensor) -> float:
        """
        Score each architecture with predefined architecture and data
        :param arch: architecture to be scored
        :param pre_defined: pre-defined evaluation args
        :param batch_data: a mini batch of data, [ batch_size, channel, W, H ]
        :param batch_labels: a mini batch of labels
        :return: score
        """
        raise NotImplementedError

    def evaluate_wrapper(self, arch, pre_defined,
                         batch_data: torch.tensor, batch_labels: torch.tensor) -> (float, float, str):
        """
        :param arch: architecture to be scored
        :param pre_defined: pre-defined evaluation args
        :param batch_data: a mini batch of data, [ batch_size, channel, W, H ]
        :param batch_labels: a mini batch of labels
        :return: score, timeUsage
        """

        # torch.cuda.empty_cache()

        start = time.time()
        arch.train()
        arch.zero_grad()
        # gpu_util_begin = showUtilization()

        score = self.evaluate(arch, pre_defined, batch_data, batch_labels)

        torch.cuda.empty_cache()

        # gpu_util_end = showUtilization()
        end = time.time()

        # assume there are 3 devices, 0, 1, 2
        # if "cpu" in pre_defined.device:
        #     gpu_res = "cpu usage"
        # elif "0" in pre_defined.device:
        #     begin_str = gpu_util_begin[0]["MEM"]
        #     end_str = gpu_util_end[0]["MEM"]
        #     gpu_res = end_str + "-" + begin_str
        # elif "1" in pre_defined.device:
        #     begin_str = gpu_util_begin[1]["MEM"]
        #     end_str = gpu_util_end[1]["MEM"]
        #     gpu_res = end_str + "-" + begin_str
        # elif "2" in pre_defined.device:
        #     begin_str = gpu_util_begin[2]["MEM"]
        #     end_str = gpu_util_end[2]["MEM"]
        #     gpu_res = end_str + "-" + begin_str
        # else:
        #     logger.info("GPU id not found in GPU util class")
        gpu_res = "NotImplemented"

        return score, end-start, gpu_res
