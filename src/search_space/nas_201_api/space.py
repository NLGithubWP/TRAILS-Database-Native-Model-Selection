

from nas_201_api import NASBench201API
from search_space.core.space import SpaceWrapper
from search_space.nas_201_api.lib import nasbench2
from search_space.nas_201_api.lib.nasbench2 import get_arch_str_from_model
from search_space.nas_201_api.model_params import NasBench201Cfg
from search_space.utils.weight_initializers import init_net


class NasBench201Space(SpaceWrapper):

    def __init__(self, api_loc: str, modelCfg: NasBench201Cfg):
        super().__init__(modelCfg)
        self.api = NASBench201API(api_loc)

    def new_architecture(self, arch_id: int = 0):
        return self[arch_id]

    def new_architecture_hash(self, arch_hash: str):
        architecture = nasbench2.get_model_from_arch_str(arch_hash, self.model_cfg.num_labels, self.model_cfg.bn)
        init_net(architecture, self.model_cfg.init_w_type, self.model_cfg.init_b_type)
        return architecture

    def query_performance(self, arch_id: int) -> dict:

        dataset = ""
        if self.model_cfg.dataset_name == "cifar10":
            dataset = "cifar10-valid"
        else:
            dataset = self.model_cfg.dataset_name

        info = self.api.get_more_info(int(arch_id), dataset, iepoch=None, hp='200', is_random=False)
        static = {
            "architecture_id": arch_id,
            "trainable_parameters": None,
            "training_time": info['train-per-time'],
            "train_accuracy": info['train-accuracy'],
            "validation_accuracy": info['valid-accuracy'],
            "test_accuracy": info['test-accuracy'],
        }

        return static

    def query_performance_hash(self, arch_str: str) -> dict:
        return self.query_performance(int(arch_str))

    def __getitem__(self, index):
        arch_str = self.api[index]
        architecture = nasbench2.get_model_from_arch_str(arch_str, self.model_cfg.num_labels, self.model_cfg.bn)
        init_net(architecture, self.model_cfg.init_w_type, self.model_cfg.init_b_type)
        return architecture

    def __len__(self):
        return len(self.api)

    def get_size(self, architecture) -> int:
        arch_str = get_arch_str_from_model(architecture)
        return len([ele for ele in arch_str.split("|") if "none" not in ele])

