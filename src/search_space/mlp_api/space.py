
import random
import time
from copy import deepcopy
from torch import optim
from common.constant import Config, CommonVars
from eva_engine import evaluator_register
from search_space.core.model_params import ModelMicroCfg, ModelMacroCfg
from search_space.core.space import SpaceWrapper
from search_space.mlp_api.model_params import MlpMacroCfg
import torch.nn as nn
from torch.utils.data import DataLoader
import query_api.query_model_gt_acc_api as gt_api

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

    def load(self):
        pass

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

    def profiling(self, dataset: str, dataloader: DataLoader = None, device: str = None, args=None) -> (float, float):
        # pick the largest net to train
        super_net = MLP(
                  ninput=self.model_cfg.init_channels,
                  hidden_layer_list=[DEFAULT_LAYER_CHOICES_20[-1]] * self.model_cfg.num_layers,
                  dropout_rate=0,
                  noutput=self.model_cfg.num_labels)

        # get a random batch.
        for batch_idx, batch in enumerate(dataloader):

            target = batch['y']
            batch['id'] = batch['id'].to(device)
            batch['value'] = batch['value'].to(device)
            target = target.to(device)
            # .reshape(target.shape[0], self.model_cfg.num_labels).

            # measure score time,
            score_time_begin = time.time()
            naswot_score, _ = evaluator_register[CommonVars.NAS_WOT].evaluate_wrapper(
                arch=super_net,
                device=device,
                batch_data=batch['value'],
                batch_labels=target)

            score_time = time.time() - score_time_begin

            # re-init hte net
            del super_net
            super_net = MLP(
                ninput=self.model_cfg.init_channels,
                hidden_layer_list=[DEFAULT_LAYER_CHOICES_20[-1]] * self.model_cfg.num_layers,
                dropout_rate=0,
                noutput=self.model_cfg.num_labels)

            # optimizer
            opt_metric = nn.CrossEntropyLoss(reduction='mean')
            opt_metric = opt_metric.to(device)
            optimizer = optim.Adam(super_net.parameters(), lr=args.lr)

            # measure training for one epoch time
            train_time_begin = time.time()
            y = super_net(batch['value'])
            loss = opt_metric(y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_time_iter = time.time() - train_time_begin

            # todo: this is pre-defined by using img Dataset, suppose each epoch only train 200 iterations
            score_time_per_model = score_time
            train_time_per_epoch = train_time_iter * 200
            N_K_ratio = gt_api.profile_NK_trade_off(dataset)
            return score_time_per_model, train_time_per_epoch, N_K_ratio

    def micro_to_id(self, arch_struct: ModelMicroCfg) -> str:
        assert isinstance(arch_struct, MlpMicroCfg)
        return str(arch_struct.hidden_layer_list)

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
        return len(DEFAULT_LAYER_CHOICES_20) ** self.model_cfg.num_layers

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
            layer_size = random.choice(DEFAULT_LAYER_CHOICES_20)
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
            cur_layer_size = child_layer_list[chosen_hidden_layer_index]
            cur_index = DEFAULT_LAYER_CHOICES_20.index(cur_layer_size)

            # right traverse
            if cur_index + 1 <= len(DEFAULT_LAYER_CHOICES_20) - 1:
                new_model = MlpMicroCfg(child_layer_list)
                return str(new_model), new_model

            # left traverse
            elif cur_index - 1 <= len(DEFAULT_LAYER_CHOICES_20)-1:
                new_model = MlpMicroCfg(child_layer_list)
                return str(new_model), new_model
            else:
                raise Exception


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


