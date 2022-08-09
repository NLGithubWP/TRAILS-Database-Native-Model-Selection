

import collections
from third_party.models import CellStructure
import random
from sampler.core.sample import Sampler
from search_space import SpaceWrapper


class Model(object):
    def __init__(self):
        self.arch = None
        self.score = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return "{:}".format(self.arch)


class RegularizedEASampler(Sampler):

    def __init__(self, space: SpaceWrapper, args):
        super().__init__(space)

        self.population_size = args.population_size
        self.population = collections.deque()
        self.space = space
        self.sample_size = args.sample_size
        self.current_sampled = 0
        self.current_arch_structure = None

    def sample_next_arch(self, max_nodes: int) -> (str, CellStructure):
        """
        # Carry out evolution in cycles. Each cycle produces a model and removes another
        # Sample randomly chosen models from the current population.
        # Inefficient, but written this way for clarity. In the case of neural
        # nets, the efficiency of this line is irrelevant because training neural
        # nets is the rate-determining step.
        :param max_nodes:
        :return:
        """
        while True:
            # Initialize the population with random models.
            if len(self.population) < self.population_size:
                while True:
                    arch_id, arch_struc = self.space.random_architecture_id(max_nodes)
                    # make sure EA population has no repeated value
                    if arch_id not in self.population:
                        self.current_arch_structure = arch_struc
                        yield str(arch_id), arch_struc
            else:
                sample = []
                while len(sample) < self.sample_size:
                    candidate = random.choice(list(self.population))
                    sample.append(candidate)
                # The parent is the model with best score in the sample.
                parent = max(sample, key=lambda i: i.score)
                arch_struct = self.space.mutate_architecture(parent.arch)
                arch_id = self.space.arch_to_id(arch_struct)
                self.current_arch_structure = arch_struct
                yield str(arch_id), arch_struct

    def fit_sampler(self, score: float):
        # if it;s in Initialize stage, add to the population with random models.
        if len(self.population) < self.population_size:
            model = Model()
            model.arch = self.current_arch_structure
            model.score = score
            self.population.append(model)

        # if it's in mutation stage
        else:
            child = Model()
            child.arch = self.current_arch_structure
            child.score = score

            self.population.append(child)
            # Remove the oldest model.
            self.population.popleft()
