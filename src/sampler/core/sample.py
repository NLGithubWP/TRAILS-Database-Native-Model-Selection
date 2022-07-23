

from abc import abstractmethod
from torch import nn
from search_space import SpaceWrapper


class Sampler:

    @abstractmethod
    def sample_next_arch(self, space: SpaceWrapper, required_size: int = 1) -> (int, nn.Module):
        raise NotImplementedError


