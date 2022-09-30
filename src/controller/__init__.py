

from common.constant import CommonVars
from controller.sampler_EA.regularized_ea import RegularizedEASampler
from controller.sampler_all.seq_sampler import SequenceSampler
from controller.sampler_RL.reinforcement_learning import RLSampler
from controller.sampler_RN.random_sample import RandomSampler
from controller.sampler_all.seq_sampler import SequenceSampler

sampler_register = {
    CommonVars.TEST_SAMPLER: SequenceSampler,
    # CommonVars.RANDOM_SAMPLER: RandomSampler,
    CommonVars.RANDOM_SAMPLER: SequenceSampler,
    CommonVars.RL_SAMPLER: RLSampler,
    CommonVars.EA_SAMPLER: RegularizedEASampler,
}

