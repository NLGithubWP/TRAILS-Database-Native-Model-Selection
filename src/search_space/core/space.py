
from abc import abstractmethod
from torch import nn
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

    # def copy_architecture(self, arch_id: str, architecture: nn.Module, new_bn: bool = True) -> nn.Module:
    #     """
    #     Copy an architecture by id, it creates an new architecture, and then load the static dict
    #     :param arch_id:
    #     :param architecture:
    #     :return:
    #     """
    #     # arch_new_time = time.time()
    #     self.update_bn_flag(new_bn)
    #     new_architecture = self.new_architecture(arch_id)
    #     # print("--in copy arch, time to new arch = " + str(time.time() - arch_new_time))
    #     # arch_load_time = time.time()
    #     # new_architecture.load_state_dict(architecture.state_dict(), strict=False)
    #     # print("--in copy arch, time to load dic = " + str(time.time() - arch_load_time))
    #     return new_architecture
