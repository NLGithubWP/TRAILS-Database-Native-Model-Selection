import json
from copy import deepcopy
from common.constant import Config
from search_space.core.model_params import ModelMicroCfg, ModelMacroCfg
from search_space.nas_201_api.rl_policy import RLPolicy201Topology
from third_party.models import get_search_spaces, CellStructure
from search_space.core.space import SpaceWrapper
from third_party.sp201_lib import nasbench2, NASBench201API
from third_party.sp201_lib.nasbench2 import get_arch_str_from_model
from search_space.nas_201_api.model_params import NB201MacroCfg
from search_space.utils.weight_initializers import init_net
import random
import ConfigSpace
import query_api.query_model_gt_acc_api as gt_api
from torch.utils.data import DataLoader


class NB201MicroCfg(ModelMicroCfg):

    @classmethod
    def builder(cls, encoding: str):
        data = json.loads(encoding)
        return NB201MicroCfg(CellStructure.str2structure(data["cell_str"]), data["arch_hash"])

    def __init__(self, cell_structure: CellStructure, arch_hash: str):
        super().__init__()
        self.cell_struct = cell_structure
        self.arch_hash = arch_hash

    def __str__(self):
        return json.dumps({
            "cell_str": self.cell_struct.tostr(),
            "arch_hash": self.arch_hash})


class NasBench201Space(SpaceWrapper):

    def __init__(self, api_loc: str, modelCfg: NB201MacroCfg):
        super().__init__(modelCfg, Config.NB201)
        self.api_loc = api_loc
        self.api = NASBench201API(api_loc)

    def load(self):
        self.api = NASBench201API(self.api_loc)

    @classmethod
    def serialize_model_encoding(cls, arch_micro: ModelMicroCfg) -> str:
        assert isinstance(arch_micro, NB201MicroCfg)
        return str(arch_micro)

    @classmethod
    def deserialize_model_encoding(cls, model_encoding: str) -> ModelMicroCfg:
        return NB201MicroCfg.builder(model_encoding)

    @classmethod
    def new_arch_scratch(cls, arch_macro: ModelMacroCfg, arch_micro: ModelMicroCfg, bn: bool = True):
        assert isinstance(arch_micro, NB201MicroCfg)
        assert isinstance(arch_macro, NB201MacroCfg)

        architecture = nasbench2.get_model_from_arch_str(
            arch_micro.arch_hash,
            arch_macro.num_labels,
            arch_macro.bn,
            arch_macro.init_channels)
        init_net(architecture, arch_macro.init_w_type, arch_macro.init_b_type)
        return architecture

    def micro_to_id(self, arch_struct: ModelMicroCfg) -> str:
        assert isinstance(arch_struct, NB201MicroCfg)
        arch_id = self.api.query_index_by_arch(arch_struct.cell_struct)
        return str(arch_id)

    def profiling(self, dataset: str, dataloader: DataLoader = None, args=None) -> (float, float, int):
        score_time_per_model = gt_api.guess_score_time(self.name, dataset)
        train_time_per_epoch = gt_api.guess_train_one_epoch_time(self.name, dataset)
        N_K_ratio = gt_api.profile_NK_trade_off(dataset)
        return score_time_per_model, train_time_per_epoch, N_K_ratio

    def new_architecture(self, arch_id: str):
        arch_hash = self.api[int(arch_id)]

        architecture = nasbench2.get_model_from_arch_str(
            arch_hash,
            self.model_cfg.num_labels,
            self.model_cfg.bn,
            self.model_cfg.init_channels)

        init_net(architecture, self.model_cfg.init_w_type, self.model_cfg.init_b_type)
        return architecture

    def new_architecture_with_micro_cfg(self, arch_micro: ModelMicroCfg):
        assert isinstance(arch_micro, NB201MicroCfg)

        # micro => id => has => arch
        arch_id = self.api.query_index_by_arch(arch_micro.cell_struct)
        arch_hash = self.api[int(arch_id)]

        architecture = nasbench2.get_model_from_arch_str(
            arch_hash,
            self.model_cfg.num_labels,
            self.model_cfg.bn,
            self.model_cfg.init_channels)

        init_net(architecture, self.model_cfg.init_w_type, self.model_cfg.init_b_type)
        return architecture

    def __len__(self):
        return len(self.api)

    def get_arch_size(self, arch_micro) -> int:
        arch_str = get_arch_str_from_model(arch_micro)
        return len([ele for ele in arch_str.split("|") if "none" not in ele])

    def random_architecture_id(self, max_nodes: int) -> (str, ModelMicroCfg):
        """
        default 4 nodes in 201
        :param max_nodes:
        :return:
        """
        max_nodes = 4
        op_names = get_search_spaces("tss", "nats-bench")
        while True:
            genotypes = []
            for i in range(1, max_nodes):
                xlist = []
                for j in range(i):
                    node_str = "{:}<-{:}".format(i, j)
                    op_name = random.choice(op_names)
                    xlist.append((op_name, j))
                genotypes.append(tuple(xlist))

            arch_struc = CellStructure(genotypes)
            arch_id = self.api.query_index_by_arch(arch_struc)
            arch_hash = self.api[int(arch_id)]
            if arch_id != -1:
                return str(arch_id), NB201MicroCfg(arch_struc, arch_hash)

    '''Below is for EA'''
    def mutate_architecture(self, parent_arch: ModelMicroCfg) -> (str, ModelMicroCfg):
        """Computes the architecture for a child of the given parent architecture.
        The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
        """
        assert isinstance(parent_arch, NB201MicroCfg)
        op_names = get_search_spaces("tss", "nats-bench")

        child_arch = deepcopy(parent_arch.cell_struct)
        node_id = random.randint(0, len(child_arch.nodes) - 1)
        node_info = list(child_arch.nodes[node_id])
        snode_id = random.randint(0, len(node_info) - 1)
        xop = random.choice(op_names)
        while xop == node_info[snode_id][0]:
            xop = random.choice(op_names)
        node_info[snode_id] = (xop, node_info[snode_id][1])
        child_arch.nodes[node_id] = tuple(node_info)

        arch_struc = CellStructure(child_arch)
        arch_id = self.api.query_index_by_arch(arch_struc)
        arch_hash = self.api[int(arch_id)]
        return arch_id, NB201MicroCfg(child_arch, arch_hash)

    '''Below is for RL and BOHB'''

    def get_reinforcement_learning_policy(self, rl_learning_rate):
        op_names = get_search_spaces("tss", "nats-bench")
        return RLPolicy201Topology(op_names, rl_learning_rate)

    def get_configuration_space(self):
        max_nodes = 4
        op_names = get_search_spaces("tss", "nats-bench")
        cs = ConfigSpace.ConfigurationSpace()
        # edge2index   = {}
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                cs.add_hyperparameter(
                    ConfigSpace.CategoricalHyperparameter(node_str, op_names)
                )
        return cs

    def config2arch_func(self, config):
        max_nodes = 4
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = config[node_str]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)
