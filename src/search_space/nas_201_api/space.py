
from copy import deepcopy
from common.constant import Config
from search_space.nas_201_api.rl_policy import RLPolicy201Topology
from third_party.models import get_search_spaces, CellStructure
from search_space.core.space import SpaceWrapper
from search_space.nas_201_api.lib import nasbench2, NASBench201API
from search_space.nas_201_api.lib.nasbench2 import get_arch_str_from_model
from search_space.nas_201_api.model_params import NasBench201Cfg
from search_space.utils.weight_initializers import init_net
import random
import ConfigSpace


class NasBench201Space(SpaceWrapper):

    def __init__(self, api_loc: str, modelCfg: NasBench201Cfg):
        super().__init__(modelCfg, Config.NB201)
        self.api = NASBench201API(api_loc)

    def new_architecture(self, arch_id: str):
        arch_str = self.api[int(arch_id)]
        return self.new_architecture_hash(arch_str)

    def new_architecture_hash(self, arch_hash: str):
        architecture = nasbench2.get_model_from_arch_str(
            arch_hash,
            self.model_cfg.num_labels,
            self.model_cfg.bn,
            self.model_cfg.init_channels)

        init_net(architecture, self.model_cfg.init_w_type, self.model_cfg.init_b_type)
        return architecture

    def __len__(self):
        return len(self.api)

    def archid_to_hash(self, arch_id):
        return self.api[int(arch_id)]

    def get_arch_size(self, architecture) -> int:
        arch_str = get_arch_str_from_model(architecture)
        return len([ele for ele in arch_str.split("|") if "none" not in ele])

    def random_architecture_id(self, max_nodes: int) -> (str, object):
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
            if arch_id != -1:
                return str(arch_id), arch_struc

    def mutate_architecture(self, parent_arch: object) -> object:
        """Computes the architecture for a child of the given parent architecture.
        The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
        """
        assert isinstance(parent_arch, CellStructure)
        op_names = get_search_spaces("tss", "nats-bench")

        child_arch = deepcopy(parent_arch)
        node_id = random.randint(0, len(child_arch.nodes) - 1)
        node_info = list(child_arch.nodes[node_id])
        snode_id = random.randint(0, len(node_info) - 1)
        xop = random.choice(op_names)
        while xop == node_info[snode_id][0]:
            xop = random.choice(op_names)
        node_info[snode_id] = (xop, node_info[snode_id][1])
        child_arch.nodes[node_id] = tuple(node_info)
        return child_arch

    def get_reinforcement_learning_policy(self, rl_learning_rate):
        op_names = get_search_spaces("tss", "nats-bench")
        return RLPolicy201Topology(op_names, rl_learning_rate)

    def arch_to_id(self, arch_struct: object) -> str:
        assert isinstance(arch_struct, CellStructure)
        arch_id = self.api.query_index_by_arch(arch_struct)
        return str(arch_id)

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
