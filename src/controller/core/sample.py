

from abc import abstractmethod
from search_space import SpaceWrapper
from third_party.models import CellStructure


class Sampler:

    def __init__(self, space: SpaceWrapper):
        self.space = space

    @abstractmethod
    def sample_next_arch(self, max_nodes: int) -> (str, CellStructure):
        """
        Sample next architecture,
        :param space: search space wrapper
        :param max_nodes: how many nodes in each cell.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def fit_sampler(self, score: float):
        """
        Fit the sampler with architecture's score.
        :param score:
        :return:
        """
        raise NotImplementedError

