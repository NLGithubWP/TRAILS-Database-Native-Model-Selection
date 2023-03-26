

import collections
from controller.core.sample import Sampler
from search_space.core.model_params import ModelMicroCfg
from controller.core.sample import Sampler
import random

from search_space.core.space import SpaceWrapper


class Model(object):
    def __init__(self):
        self.arch = None
        self.score = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return "{:}".format(self.arch)


class RegularizedEASampler(Sampler):

    def __init__(self, space: SpaceWrapper, population_size: int, sample_size: int):
        super().__init__(space)

        self.population_size = population_size
        self.population = collections.deque()
        self.space = space
        self.sample_size = sample_size
        self.current_sampled = 0
        self.current_arch_micro = None

        # todo: visited, collarpaze

    def sample_next_arch(self, sorted_model: list) -> (str, ModelMicroCfg):
        """
        # Carry out evolution in cycles. Each cycle produces a model and removes another
        # Sample randomly chosen models from the current population.
        # Inefficient, but written this way for clarity. In the case of neural
        # nets, the efficiency of this line is irrelevant because training neural
        # nets is the rate-determining step.
        :param sorted_model:
        :return:
        """
        # Initialize the population with random models.
        if len(self.population) < self.population_size:
            while True:
                arch_id, arch_micro = self.space.random_architecture_id()
                # make sure EA population has no repeated value
                if arch_id not in self.population:
                    break
            self.current_arch_micro = arch_micro
            return arch_id, arch_micro
        else:
            sample = []
            while len(sample) < self.sample_size:
                candidate = random.choice(list(self.population))
                sample.append(candidate)
            # The parent is the model with best score in the sample.
            # todo: here use prue score may not accurate since the score is dynamic in the fly.
            parent = max(sample, key=lambda i: sorted_model.index(str(i.arch)))
            # parent = max(sample, key=lambda i: i.score)
            arch_id, arch_micro = self.space.mutate_architecture(parent.arch)
            self.current_arch_micro = arch_micro
            return arch_id, arch_micro

    def fit_sampler(self, score: float):
        # if it's in Initialize stage, add to the population with random models.
        if len(self.population) < self.population_size:
            model = Model()
            model.arch = self.current_arch_micro
            model.score = score
            self.population.append(model)

        # if it's in mutation stage
        else:
            child = Model()
            child.arch = self.current_arch_micro
            child.score = score

            self.population.append(child)
            # Remove the oldest model.
            self.population.popleft()
