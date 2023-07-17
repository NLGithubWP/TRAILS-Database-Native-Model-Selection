

##################################################
import os, sys, torch, random, PIL, copy, numpy as np
from os import path as osp
from shutil import copyfile


def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

