

from controller.core.sample import Sampler
from search_space import SpaceWrapper
from third_party.models import CellStructure


class RandomSampler(Sampler):

    def __init__(self, space: SpaceWrapper, args):
        super().__init__(space)
        self.visited = []

    def sample_next_arch(self, max_nodes: int) -> (str, CellStructure):
        while True:
            arch_id, arch_struc = self.space.random_architecture_id(max_nodes)

            if arch_id not in self.visited:
                self.visited.append(arch_id)
                yield str(arch_id), arch_struc

    def fit_sampler(self, score: float):
        # random sampler can skip this.
        pass
