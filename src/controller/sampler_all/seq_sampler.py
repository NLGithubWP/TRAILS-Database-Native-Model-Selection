
import random

from controller.core.sample import Sampler
from controller.core.sample import Sampler
from search_space.core.model_params import ModelMicroCfg
from search_space.core.space import SpaceWrapper
from third_party.models import CellStructure


class SequenceSampler(Sampler):

    def __init__(self, space: SpaceWrapper, args):
        super().__init__(space)

    def sample_next_arch(self, sorted_model: list) -> (str, ModelMicroCfg):
        """
        Sample one random architecture, can sample max 10k architectures.
        :return: arch_id, architecture
        """
        # random.seed(20)
        arch_id_list = self.space.sample_all_models()
        for arch_id in arch_id_list:
            yield str(arch_id), None

    def fit_sampler(self, score: float):
        pass





