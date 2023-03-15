

from abc import abstractmethod

from search_space.core.model_params import ModelMicroCfg
from search_space.core.space import SpaceWrapper


class Sampler:

    def __init__(self, space: SpaceWrapper):
        self.space = space

    @abstractmethod
    def sample_next_arch(self, max_nodes: int) -> (str, ModelMicroCfg):
        """
        Sample next architecture,
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

