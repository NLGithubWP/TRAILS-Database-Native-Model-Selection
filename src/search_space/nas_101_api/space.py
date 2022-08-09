import copy
import itertools
import random

import ConfigSpace
import numpy as np

from logger import logger
from search_space.core.space import SpaceWrapper
from search_space.nas_101_api.lib import nb101_api
from search_space.nas_101_api.lib.model import NasBench101Network
from search_space.nas_101_api.lib.nb101_api import ModelSpec
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

    def __init__(self, api_loc: str, modelCfg: NasBench101Cfg):
        super().__init__(modelCfg)
        self.api = nb101_api.NASBench(api_loc)

    def new_architecture(self, arch_id: int):
        arch_hash = next(itertools.islice(self.api.hash_iterator(), arch_id, None))
        return self.new_architecture_hash(arch_hash)

    def new_architecture_hash(self, arch_hash: str):
        spec = self._get_spec(arch_hash)
        # generate network with adjacency and operation
        architecture = NasBench101Network(spec, self.model_cfg)
        return architecture

    def query_performance(self, arch_id: int, dataset_name: str) -> dict:

        if dataset_name != "cifar10":
            logger.info("NasBench101 only be evaluated at CIFAR10")

        arch_hash = next(itertools.islice(self.api.hash_iterator(), arch_id, None))
        return self.query_performance_hash(arch_hash, dataset_name)

    def query_performance_hash(self, arch_hash: str, dataset_name: str) -> dict:

        if dataset_name != "cifar10":
            logger.info("NasBench101 only be evaluated at CIFAR10")

        res = self.api.query(self._get_spec(arch_hash))
        static = {
            "architecture_id": arch_hash,
            "trainable_parameters": res["trainable_parameters"],
            "training_time": res["training_time"],
            "train_accuracy": res["train_accuracy"],
            "validation_accuracy": res["validation_accuracy"],
            "test_accuracy": res["test_accuracy"],
        }

        # this result repeated three times.
        # spec = self._get_spec(arch_id)
        # _, stats2 = self.api.get_metrics_from_spec(spec)
        return static

    def __len__(self):
        return len(self.api.hash_iterator())

    def get_arch_size(self, architecture) -> int:
        return len(architecture.spec.matrix)

    def _get_spec(self, arch_hash: str):
        matrix = self.api.fixed_statistics[arch_hash]['module_adjacency']
        operations = self.api.fixed_statistics[arch_hash]['module_operations']
        spec = ModelSpec(matrix, operations)

        return spec

    def random_architecture_id(self, max_nodes: int) -> (int, object):
        """Returns a random valid spec."""
        while True:
            matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = ModelSpec(matrix=matrix, ops=ops)
            if self.api.is_valid(spec):
                arch_id = self.arch_to_id(spec)
                return arch_id, spec

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
            if self.api.is_valid(new_spec):
                return new_spec

    def get_reinforcement_learning_policy(self, rl_learning_rate):
        return RLPolicy101Topology(self, rl_learning_rate, NUM_VERTICES)

    def arch_to_id(self, arch_spec: object) -> int:
        assert isinstance(arch_spec, ModelSpec)
        if self.api.is_valid(arch_spec):
            arch_hash = arch_spec.hash_spec(ALLOWED_OPS)
            arch_id = list(self.api.hash_iterator()).index(arch_hash)
            return arch_id
        else:
            return -1

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

    a = NasBench101Space(api_loc, model_cfg)

    aid, spec = a.random_architecture_id(4)
    a.mutate_architecture(spec)


