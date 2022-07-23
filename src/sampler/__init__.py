from common.constant import CommonVars
from sampler.random_sample import RandomSampler
from sampler.seq_sampler import SequenceSampler


sampler_register = {
    CommonVars.TEST_SAMPLER: SequenceSampler(),
    CommonVars.RANDOM_SAMPLER: RandomSampler()

}

