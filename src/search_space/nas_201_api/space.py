

from nas_201_api import NASBench201API
from search_space.core.space import SpaceWrapper
from search_space.nas_201_api.lib import nasbench2
from search_space.nas_201_api.model_params import NasBench201Cfg
from search_space.utils.weight_initializers import init_net


class NasBench201Space(SpaceWrapper):

    def __init__(self, api_loc: str, modelCfg: NasBench201Cfg):
        super().__init__(modelCfg)
        self.api = NASBench201API(api_loc)

    def new_architecture(self, arch_size: int = 0):
        for arch_id, arch_str in enumerate(self.api):
            architecture = nasbench2.get_model_from_arch_str(arch_str, self.model_cfg.num_labels)
            init_net(architecture, self.model_cfg.init_w_type, self.model_cfg.init_b_type)
            yield arch_id, architecture

    def query_performance(self, arch_id: str) -> dict:

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

    def __getitem__(self, index):
        return self.api[index]

    def __len__(self):
        return len(self.api)

    def __iter__(self):
        pass

