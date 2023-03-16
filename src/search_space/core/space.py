
from abc import abstractmethod
from torch.utils.data import DataLoader
from search_space.core.model_params import ModelMacroCfg, ModelMicroCfg


class SpaceWrapper:

    def __init__(self, cfg: ModelMacroCfg, name: str):
        self.model_cfg = cfg
        self.name = name

    """serialize and deserialize"""
    @classmethod
    def serialize_model_encoding(cls, arch_micro: ModelMicroCfg) -> str:
        raise NotImplementedError

    @classmethod
    def deserialize_model_encoding(cls, model_encoding) -> ModelMicroCfg:
        raise NotImplementedError

    @classmethod
    def new_arch_scratch(cls, arch_macro: ModelMacroCfg, arch_micro: ModelMicroCfg):
        """
        Args:
            arch_macro: macro setting for one architecture
            arch_micro: micro setting for one architecture
        Returns:
        """
        raise NotImplementedError

    @abstractmethod
    def profiling(self, dataset: str, dataloader: DataLoader = None, device: str = None, args=None) -> (float, float):
        """
        Profile the training and scoring time.
        Args:
            args:
            device:
            dataset:
            dataloader:
        Returns:
        """
        raise NotImplementedError

    @abstractmethod
    def micro_to_id(self, arch_struct: ModelMicroCfg) -> str:
        raise NotImplementedError

    """init new architecture"""
    @abstractmethod
    def new_architecture(self, arch_id: str):
        """
        Generate an architecture with arch id
        :return:
        """
        raise NotImplementedError

    def new_architecture_with_micro_cfg(self, arch_micro: ModelMicroCfg):
        """
        Generate an architecture with arch_micro
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
    def get_arch_size(self, architecture):
        """
        Get how many edges in each cell of the architecture.
        :return:
        """
        raise NotImplementedError

    def update_bn_flag(self, bn: bool):
        """
        Update architecture's bn,
        :param bn:
        :return:
        """
        self.model_cfg.bn = bn

    """Below is for integrating space with various sampler"""

    def random_architecture_id(self, max_nodes: int) -> (str, ModelMicroCfg):
        """
        Random generate architecture id, cell structure, supporting RN, RL, R
        :param max_nodes:  how many nodes in this cell
        :return:
        """
        raise NotImplementedError

    def mutate_architecture(self, parent_arch: ModelMicroCfg) -> (str, ModelMicroCfg):
        """
        Mutate architecture, this is to support EA sampler
        :rtype: object
        :return:
        """
        raise NotImplementedError

    def get_reinforcement_learning_policy(self, lr_rate):
        """
        This is fpr reinforcement learning policy sampler
        :return:
        """
        raise NotImplementedError
