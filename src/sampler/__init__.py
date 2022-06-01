
from common.constant import RANDOM_SAMPLER
from sampler.random_sample import RandomSampler

sampler_register = {}

sampler_register[RANDOM_SAMPLER] = RandomSampler()
