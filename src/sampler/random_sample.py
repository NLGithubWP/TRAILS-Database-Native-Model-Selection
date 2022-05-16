import numpy as np

from sampler.core.sample import Sampler
from search_space.NASBench101.model_space import *
from search_space.NASBench101.network import NasBench101Network
from search_space.core.network import Architecture


class RandomSampler(Sampler):

    def __init__(self):
        super(RandomSampler, self).__init__()

    def sample_one_arch(self, space, args, query_apis) -> (Architecture, list, list):

        adjacency_matrix, ops = NASBench101ModelSpec.sample_random_architecture(query_apis)

        print(adjacency_matrix)
        print(ops)

        print("sampling architecture from space = ", space)

        # matrix = [[0, 1, 1, 1, 0, 1, 0],
        #           [0, 0, 0, 0, 0, 0, 1],
        #           [0, 0, 0, 0, 0, 0, 1],
        #           [0, 0, 0, 0, 1, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 1],
        #           [0, 0, 0, 0, 0, 0, 1],
        #           [0, 0, 0, 0, 0, 0, 0]]
        # operations = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3',
        #               'output']
        net = NasBench101Network(adjacency_matrix, ops, args)
        # print(net)
        return net, adjacency_matrix, ops













