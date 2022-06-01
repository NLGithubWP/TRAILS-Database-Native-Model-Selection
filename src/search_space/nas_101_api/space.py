
import itertools
from nas_101_api.model_spec import ModelSpec
from search_space.core.space import SpaceWrapper
from search_space.nas_101_api.lib import nb101_api
from search_space.nas_101_api.lib.model import NasBench101Network
from search_space.nas_101_api.model_params import NasBench101Cfg


class NasBench101Space(SpaceWrapper):

    def __init__(self, api_loc: str, modelCfg: NasBench101Cfg):
        super().__init__(modelCfg)
        self.api = nb101_api.NASBench(api_loc)

    def new_architecture(self, arch_size: int):
        for arch_id in self.api.hash_iterator():
            # get adjacency and operation using arch_id
            spec = self._get_spec(arch_id)
            # generate network with adjacency and operation
            architecture = NasBench101Network(spec, self.model_cfg)

            if len(architecture.spec.matrix) >= arch_size:
                yield arch_id, architecture

    def query_performance(self, arch_id: str) -> dict:
        res = self.api.query(self._get_spec(arch_id))
        static = {
            "architecture_id": arch_id,
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

    def __iter__(self):
        for arch_id in self.api.hash_iterator():
            spec = self._get_spec(arch_id)
            # generate network with adjacency and operation
            architecture = NasBench101Network(spec, self.model_cfg)
            yield arch_id, architecture

    def __getitem__(self, index):
        return next(itertools.islice(self.api.hash_iterator(), index, None))

    def __len__(self):
        return len(self.api.hash_iterator())

    def _get_spec(self, arch_id: str):
        matrix = self.api.fixed_statistics[arch_id]['module_adjacency']
        operations = self.api.fixed_statistics[arch_id]['module_operations']
        spec = ModelSpec(matrix, operations)
        return spec

