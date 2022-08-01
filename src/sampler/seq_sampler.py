
import random
from torch import nn
from logger import logger
from sampler.core.sample import Sampler
from search_space import SpaceWrapper


class SequenceSampler(Sampler):

    def __init__(self):
        super(SequenceSampler, self).__init__()

    def sample_next_arch(self, space: SpaceWrapper, required_size: int = 1) -> (int, nn.Module):
        """
        Sample one random architecture, can sample max 10k architectures.
        :param space: search space,
        :param required_size: how many edges the model's cell should greater than
        :return: arch_id, architecture
        """

        # For test only
        arch_id_list = [6157, 6157, 6157, 6157]

        for arch_id in arch_id_list:
            architecture = space.new_architecture(arch_id)
            if space.get_size(architecture) > required_size:
                yield arch_id, architecture
            else:
                logger.info("One arch's size " + str(arch_id) + " is smaller than the required, search next")




