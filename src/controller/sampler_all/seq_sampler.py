
import random

from controller.core.sample import Sampler
from search_space import SpaceWrapper
from third_party.models import CellStructure


class SequenceSampler(Sampler):

    def __init__(self, space: SpaceWrapper, args):
        super().__init__(space)

    def sample_next_arch(self, max_nodes: int) -> (str, CellStructure):
        """
        Sample one random architecture, can sample max 10k architectures.
        :param space: search space,
        :param required_size: how many edges the model's cell should greater than
        :return: arch_id, architecture
        """
        # random.seed(20)

        total_num_arch = len(self.space)
        arch_id_list = random.sample(range(total_num_arch), total_num_arch)
        print("arch_idList", arch_id_list[:30])
        for arch_id in arch_id_list:
            yield str(arch_id), None

    def fit_sampler(self, score: float):
        pass





