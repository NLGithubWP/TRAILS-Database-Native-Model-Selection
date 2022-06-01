
from abc import abstractmethod

from search_space.core.model_params import ModelCfgs


class SpaceWrapper:

    def __init__(self, cfg: ModelCfgs):
        self.model_cfg = cfg

    @abstractmethod
    def new_architecture(self, arch_size: int = 0):
        raise NotImplementedError

    @abstractmethod
    def query_performance(self, arch_id: str) -> dict:
        """
        Query from nas-bench dataset.
        :param arch_id: the target architecture
        :return: {accuracy, training_time, final_test_accuracy}
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, arch_index):
        """
        Return an architecture using index
        :param arch_index:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """
        How many architectures the space has
        :return:
        """
        raise NotImplementedError
