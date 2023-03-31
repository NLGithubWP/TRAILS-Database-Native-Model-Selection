
import random

from controller.core.sample import Sampler
from controller.core.sample import Sampler
from search_space.core.model_params import ModelMicroCfg
from search_space.core.space import SpaceWrapper
from third_party.models import CellStructure


class SequenceSampler(Sampler):

    def __init__(self, space: SpaceWrapper):
        super().__init__(space)

        self.arch_gene = self.space.sample_all_models()

    def sample_next_arch(self, sorted_model: list = None) -> (str, ModelMicroCfg):
        """
        Sample one random architecture, can sample max 10k architectures.
        :return: arch_id, architecture
        """
        # random.seed(20)
        try:
            arch_id, arch_micro = self.arch_gene.__next__()
            return arch_id, arch_micro
        except Exception as e:
            if "StopIteration" in str(e):
                print("the end")
                return None, None
            else:
                raise e

    def fit_sampler(self, score: float):
        pass
