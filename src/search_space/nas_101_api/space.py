
import itertools
from search_space.core.space import SpaceWrapper
from search_space.nas_101_api.lib import nb101_api
from search_space.nas_101_api.lib.model import NasBench101Network
from search_space.nas_101_api.lib.nb101_api import ModelSpec
from search_space.nas_101_api.model_params import NasBench101Cfg


class NasBench101Space(SpaceWrapper):

    def __init__(self, api_loc: str, modelCfg: NasBench101Cfg):
        super().__init__(modelCfg)
        self.api = nb101_api.NASBench(api_loc)

    def new_architecture(self, arch_id: int = 0):
        return self[arch_id]

    def new_architecture_hash(self, arch_hash: str):
        spec = self._get_spec(arch_hash)
        # generate network with adjacency and operation
        architecture = NasBench101Network(spec, self.model_cfg)
        return architecture

    def query_performance(self, arch_id: int) -> dict:

        arch_hash = next(itertools.islice(self.api.hash_iterator(), arch_id, None))

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

    def query_performance_hash(self, arch_hash: str) -> dict:
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

    def __getitem__(self, index):
        arch_hash = next(itertools.islice(self.api.hash_iterator(), index, None))
        spec = self._get_spec(arch_hash)
        # generate network with adjacency and operation
        architecture = NasBench101Network(spec, self.model_cfg)
        return architecture

    def __len__(self):
        return len(self.api.hash_iterator())

    def _get_spec(self, arch_hash: str):
        matrix = self.api.fixed_statistics[arch_hash]['module_adjacency']
        operations = self.api.fixed_statistics[arch_hash]['module_operations']
        spec = ModelSpec(matrix, operations)
        return spec

    def get_size(self, architecture) -> int:
        return len(architecture.spec.matrix)
