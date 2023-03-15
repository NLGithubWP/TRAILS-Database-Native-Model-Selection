import os
import random
from copy import deepcopy
from common.constant import Config
from search_space.core.model_params import ModelMicroCfg, ModelMacroCfg
from controller.core.sample import Sampler
from search_space.mlp_api.model_params import MlpMacroCfg
import torch.nn as nn

# Useful constants
DEFAULT_LAYER_CHOICES_20 = [8, 16, 24, 32, # 8
                            48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, # 16
                            384, 512]
DEFAULT_LAYER_CHOICES_12 = [8, 16, 24, 32,
                            48, 64, 80, 96, 112, 128, 144, 160]


class MlpMicroCfg(ModelMicroCfg):

    @classmethod
    def builder(cls, encoding: str):
        return MlpMicroCfg([int(ele) for ele in encoding.split("-")])

    def __init__(self, hidden_layer_list: list):
        super().__init__()
        self.hidden_layer_list = hidden_layer_list

    def __str__(self):
        return "-".join(str(x) for x in self.hidden_layer_list)


class MLP(nn.Module):

    def __init__(self, ninput: int, hidden_layer_list: list, dropout_rate: float, noutput: int = 1):
        super().__init__()
        """
        Args:
            ninput: number of input feature dim
            hidden_layer_list: [a,b,c..] each value is number of Neurons in corresponding hidden layer
            dropout_rate: if use drop out
            noutput: number of labels. 
        """
        layers = list()
        # 1. all hidden layers.
        for index, layer_size in enumerate(hidden_layer_list):
            layers.append(nn.Linear(ninput, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            ninput = layer_size
        # 2. last hidden layer
        if len(hidden_layer_list) == 0:
            last_hidden_layer_num = ninput
        else:
            last_hidden_layer_num = hidden_layer_list[-1]
        layers.append(nn.Linear(last_hidden_layer_num, noutput))

        # 3. generate the MLP
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:   FloatTensor B*ninput
        :return:    FloatTensor B*nouput
        """
        return self.mlp(x)


class MlpSpace(SpaceWrapper):
    def __init__(self, modelCfg: MlpMacroCfg):
        super().__init__(modelCfg, Config.MLPSP)

    @classmethod
    def serialize_model_encoding(cls, arch_micro: ModelMicroCfg) -> str:
        assert isinstance(arch_micro, MlpMicroCfg)
        return str(arch_micro)

    @classmethod
    def deserialize_model_encoding(cls, model_encoding: str) -> ModelMicroCfg:
        return MlpMicroCfg.builder(model_encoding)

    @classmethod
    def new_arch_scratch(cls, arch_macro: ModelMacroCfg, arch_micro: ModelMicroCfg):
        assert isinstance(arch_micro, MlpMicroCfg)
        assert isinstance(arch_macro, MlpMacroCfg)
        mlp = MLP(ninput=arch_macro.init_channels,
                  hidden_layer_list=arch_micro.hidden_layer_list,
                  dropout_rate=0,
                  noutput=arch_macro.num_labels)
        return mlp

    def new_architecture(self, arch_id: str):
        """
        Args:
            arch_id: arch id is the same as encoding.
        Returns:
        """
        arch_micro = MlpSpace.deserialize_model_encoding(arch_id)
        assert isinstance(arch_micro, MlpMicroCfg)
        mlp = MLP(ninput=self.model_cfg.init_channels,
                  hidden_layer_list=arch_micro.hidden_layer_list,
                  dropout_rate=0,
                  noutput=self.model_cfg.num_labels)
        return mlp

    def new_architecture_with_micro_cfg(self, arch_micro: ModelMicroCfg):
        assert isinstance(arch_micro, MlpMicroCfg)
        mlp = MLP(ninput=self.model_cfg.init_channels,
                  hidden_layer_list=arch_micro.hidden_layer_list,
                  dropout_rate=0,
                  noutput=self.model_cfg.num_labels)
        return mlp

    def __len__(self):
        return self.model_cfg.layer_choices ** self.model_cfg.num_layers

    def get_arch_size(self, arch_micro: ModelMicroCfg) -> int:
        assert isinstance(arch_micro, MlpMicroCfg)
        result = 1
        for ele in arch_micro.hidden_layer_list:
            result = result * ele
        return result

    def random_architecture_id(self, max_nodes: int = None) -> (str, ModelMicroCfg):
        """
        Args:
            max_nodes: max_nodes is not used here,
        Returns:
        """

        arch_encod = []
        for _ in range(self.model_cfg.num_layers):
            layer_size = random.choice(self.model_cfg.layer_choices)
            arch_encod.append(layer_size)

        model_micro = MlpMicroCfg(arch_encod)
        model_encoding = str(model_micro)
        return model_encoding, model_micro

    '''Below is for EA'''
    def mutate_architecture(self, parent_arch: ModelMicroCfg) -> (str, ModelMicroCfg):
        assert isinstance(parent_arch, MlpMicroCfg)

        child_layer_list = deepcopy(parent_arch.hidden_layer_list)

        # 1. choose layer index
        chosen_hidden_layer_index = random.choice(list(range(len(child_layer_list))))

        # 2. choose size of the layer index.
        while True:
            for ele in [8, -8, 16, -16, 128, -128]:
                modified_layer_size = child_layer_list[chosen_hidden_layer_index] + ele
                if modified_layer_size in DEFAULT_LAYER_CHOICES_20:
                    child_layer_list[chosen_hidden_layer_index] = child_layer_list[chosen_hidden_layer_index] + ele
                    new_model = MlpMicroCfg(child_layer_list)
                    return str(new_model), new_model


if __name__ == '__main__':
    model_cfg = MlpMacroCfg(4, 2, 1, DEFAULT_LAYER_CHOICES_20, True)
    ms = MlpSpace(model_cfg)
    encoding, model_micro = ms.random_architecture_id()
    b = ms.mutate_architecture(model_micro)

    arch = ms.new_architecture_with_micro_cfg(model_micro)
    print(ms)
    print(b)
    print(arch)

    print("Done")


