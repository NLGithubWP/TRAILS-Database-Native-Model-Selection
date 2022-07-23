
from abc import abstractmethod
from collections import OrderedDict

import torch
from torch import nn

from search_space.core.model_params import ModelCfgs


class SpaceWrapper:

    def __init__(self, cfg: ModelCfgs):
        self.model_cfg = cfg

    @abstractmethod
    def new_architecture(self, arch_id: int):
        """
        Generate an architecture with arch_index
        :return:
        """
        raise NotImplementedError

    def new_architecture_hash(self, arch_hash: str):
        """
        Generate an architecture with arch_index
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def query_performance(self, arch_id: int) -> dict:
        """
        Query from nas-bench dataset.
        :param arch_id: arch id
        :return: {accuracy, training_time, final_test_accuracy}
        """
        raise NotImplementedError

    def query_performance_hash(self, arch_hash: str) -> dict:
        """
        Query from nas-bench dataset.
        :param arch_hash: arch id
        :return: {accuracy, training_time, final_test_accuracy}
        """
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

    @abstractmethod
    def get_size(self, architecture):
        """
        Get how many edges in each cell of the architecture.
        :return:
        """
        raise NotImplementedError

    def copy_architecture(self, arch_id: int, architecture: nn.Module) -> nn.Module:
        """
        Copy an architecture by id, it creates an new architecture, and then load the static dict
        :param arch_id:
        :param architecture:
        :return:
        """
        new_architecture = self[arch_id]
        new_architecture.load_state_dict(architecture.state_dict(), strict=False)
        new_architecture.train()
        return new_architecture

    def copy_architecture_hash(self, arch_hash: str, architecture: nn.Module) -> nn.Module:
        """
        Copy an architecture by arch_hash / str, it creates a new architecture, and then load the static dict
        :param arch_hash: hash of the architecture
        :param architecture:
        :return:
        """
        new_architecture = self.new_architecture_hash(arch_hash)
        new_architecture.load_state_dict(architecture.state_dict(), strict=False)
        new_architecture.train()
        return new_architecture
