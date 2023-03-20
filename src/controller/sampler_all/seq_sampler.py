
import random

from controller.core.sample import Sampler
from controller.core.sample import Sampler
from search_space.core.space import SpaceWrapper
from third_party.models import CellStructure


class SequenceSampler(Sampler):

    def __init__(self, space: SpaceWrapper, args):
        super().__init__(space)

    def sample_next_arch(self, max_nodes: int = 0) -> (str, CellStructure):
        """
        Sample one random architecture, can sample max 10k architectures.
        :param space: search space,
        :param required_size: how many edges the model's cell should greater than
        :return: arch_id, architecture
        """
        # random.seed(20)
        arch_id_list = self.space.sample_all_models()
        for arch_id in arch_id_list:
            yield str(arch_id), None

    def fit_sampler(self, score: float):
        pass





