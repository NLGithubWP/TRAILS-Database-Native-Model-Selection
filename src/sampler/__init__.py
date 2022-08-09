

from common.constant import CommonVars
from sampler.sampler_EA.regularized_ea import RegularizedEASampler
from sampler.sampler_RL.reinforcement_learning import RLSampler
from sampler.sampler_RN.random_sample import RandomSampler
from sampler.sampler_all.seq_sampler import SequenceSampler

sampler_register = {
    CommonVars.TEST_SAMPLER: SequenceSampler,
    # CommonVars.RANDOM_SAMPLER: RandomSampler,
    CommonVars.RANDOM_SAMPLER: SequenceSampler,
    CommonVars.RL_SAMPLER: RLSampler,
    CommonVars.EA_SAMPLER: RegularizedEASampler,
}

