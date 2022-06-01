import numpy as np

from sampler.core.sample import Sampler
from search_space.core.network import Architecture


class RandomSampler(Sampler):

    def __init__(self):
        super(RandomSampler, self).__init__()

    def sample_one_arch(self, space, required_size=2) -> (Architecture, str):

        net, unique_hash = space.random_arch(required_size)

        # adjacency_matrix = \
        #           [[0, 0, 0, 1, 0, 0, 1],
        #           [0, 0, 1, 0, 0, 0, 1],
        #           [0, 0, 0, 0, 1, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 1],
        #           [0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0]]
        # ops = ['input', 'conv3x3-bn-relu', 'maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']

        # net = NasBench101Network(spec, args)
        # print(net)

        return net, unique_hash













