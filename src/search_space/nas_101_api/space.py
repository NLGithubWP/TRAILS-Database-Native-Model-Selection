import copy
import itertools
import json
import random

import ConfigSpace
import numpy as np

from common.constant import Config
from logger import logger
from query_api.query_p1_score_api import LocalApi
from search_space.core.space import SpaceWrapper
from third_party.sp101_lib import nb101_api
from third_party.sp101_lib.model import NasBench101Network
from third_party.sp101_lib.nb101_api import ModelSpec
from search_space.nas_101_api.model_params import NasBench101Cfg
from search_space.nas_101_api.rl_policy import RLPolicy101Topology


# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix


class NasBench101Space(SpaceWrapper):

    def __init__(self, api_loc: str, modelCfg: NasBench101Cfg, loapi: LocalApi):
        super().__init__(modelCfg, Config.NB101)
        self.api = nb101_api.NASBench(api_loc)
        self.loapi = loapi

    @classmethod
    def serialize_model_encoding(cls, matrix: list, operations: str) -> str:
        data = {"matrix": matrix, "operations": operations}
        return json.dumps(data)

    @classmethod
    def deserialize_model_encoding(cls, data_str) -> (list, str):
        data = json.loads(data_str)
        return data["matrix"], data["operations"]

    @staticmethod
    def new_architecture_default(matrix, operations, bn: bool, num_labels: int):
        model_cfg = NasBench101Cfg(
            bn=bn,
            init_channels=16,
            num_stacks=3,
            num_modules_per_stack=3,
            num_labels=num_labels
        )
        spec = ModelSpec(matrix, operations)
        model = NasBench101Network(spec, model_cfg)
        return model

    def new_architecture(self, arch_id: str):
        arch_hash = next(itertools.islice(self.api.hash_iterator(), int(arch_id), None))
        return self.new_architecture_hash(arch_hash)

    def new_architecture_hash(self, arch_hash: str):
        spec = self._get_spec(arch_hash)
        # generate network with adjacency and operation
        architecture = NasBench101Network(spec, self.model_cfg)
        return architecture

    def __len__(self):
        return len(self.api.hash_iterator())

    def get_arch_size(self, architecture) -> int:
        return len(architecture.spec.matrix)

    def _get_spec(self, arch_hash: str):
        matrix = self.api.fixed_statistics[arch_hash]['module_adjacency']
        operations = self.api.fixed_statistics[arch_hash]['module_operations']
        spec = ModelSpec(matrix, operations)

        return spec

    def random_architecture_id(self, max_nodes: int) -> (str, object):
        """Returns a random valid spec."""
        while True:
            matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = ModelSpec(matrix=matrix, ops=ops)
            if self.is_valid(spec):
                arch_id = self.arch_to_id(spec)
                return str(arch_id), spec

    def mutate_architecture(self, parent_arch: object) -> object:
        mutation_rate = 1.0
        assert isinstance(parent_arch, ModelSpec)
        """Computes a valid mutated spec from the old_spec."""
        while True:
            new_matrix = copy.deepcopy(parent_arch.original_matrix)
            new_ops = copy.deepcopy(parent_arch.original_ops)

            # In expectation, V edges flipped (note that most end up being pruned).
            edge_mutation_prob = mutation_rate / NUM_VERTICES
            for src in range(0, NUM_VERTICES - 1):
                for dst in range(src + 1, NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            # In expectation, one op is resampled.
            op_mutation_prob = mutation_rate / OP_SPOTS
            for ind in range(1, NUM_VERTICES - 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in self.api.config['available_ops'] if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)

            new_spec = ModelSpec(new_matrix, new_ops)
            if self.is_valid(new_spec) and self.is_scored(new_spec):
                return new_spec

    def is_valid(self, new_spec: ModelSpec):
        return self.api.is_valid(new_spec) and self.is_scored(new_spec)

    def is_scored(self, new_spec: ModelSpec):
        arch_id = self.arch_to_id(new_spec)
        return self.loapi.is_arch_inside_data(arch_id)

    def get_reinforcement_learning_policy(self, rl_learning_rate):
        return RLPolicy101Topology(self, rl_learning_rate, NUM_VERTICES)

    def arch_to_id(self, arch_spec: object) -> str:
        assert isinstance(arch_spec, ModelSpec)
        if self.api.is_valid(arch_spec):
            arch_hash = arch_spec.hash_spec(ALLOWED_OPS)
            arch_id = list(self.api.hash_iterator()).index(arch_hash)
            return str(arch_id)
        else:
            return "-1"

    def arch_hash_to_id(self, arch_hash: str) -> str:
        arch_id = list(self.api.hash_iterator()).index(arch_hash)
        return str(arch_id)

    def get_configuration_space(self):
        cs = ConfigSpace.ConfigurationSpace()
        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))
        for i in range(NUM_VERTICES * (NUM_VERTICES - 1) // 2):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, [0, 1]))
        return cs

    def config2arch_func(self, config):
        matrix = np.zeros([NUM_VERTICES, NUM_VERTICES], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        for i in range(NUM_VERTICES * (NUM_VERTICES - 1) // 2):
            row = idx[0][i]
            col = idx[1][i]
            matrix[row, col] = config["edge_%d" % i]
        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = ModelSpec(matrix, labeling)
        return model_spec


if __name__ == '__main__':
    api_loc = "/Users/kevin/project_python/Fast-AutoNAS/data/nasbench_only108.pkl"
    model_cfg = NasBench101Cfg(16,3,3,10,True)

    a = NasBench101Space(api_loc, model_cfg, None)

    aid, spec = a.random_architecture_id(4)
    a.mutate_architecture(spec)


