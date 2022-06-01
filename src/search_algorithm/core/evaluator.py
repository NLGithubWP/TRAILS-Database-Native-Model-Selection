from abc import abstractmethod
from search_space import Architecture
import torch


class Evaluator:
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, arch: Architecture, pre_defined, batch_data: torch.tensor, batch_labels: torch.tensor) -> float:
        """
        Score each architecture with predefined architecture and data
        :param arch: architecture to be scored
        :param pre_defined: pre-defined evaluation args
        :param batch_data: a mini batch of data, [ batch_size, channel, W, H ]
        :param batch_labels: a mini batch of labels
        :return: score
        """
        raise NotImplementedError
