import os
import sys
import logging
import argparse

import numpy as np
import warnings
from search_algorithm.nasi_search import NASISearch
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torchvision
import torch.backends.cudnn as cudnn

from utils import utils

from search_space.architectures.normal_network import NormalNetwork
from search_space.architectures.reduce_network import ReduceNetwork

parser = argparse.ArgumentParser("cifar")
parser.add_argument("--data", type=str, default="/Users/kevin/project_python/Fast-AutoNAS/data", help="location of the data corpus")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--steps", type=int, default=200, help="steps to optimize architecture")
parser.add_argument("--report_freq", type=float,default=10, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--init_channels", type=int,default=16, help="num of init channels")
parser.add_argument("--layers", type=int, default=9, help="total number of layers")
parser.add_argument("--search_space", choices=["normal", "reduce"], default="normal")
parser.add_argument("--cutout", action="store_true",default=False, help="use cutout")
parser.add_argument("--gumbel", action="store_true",default=False, help="use gumbel")
parser.add_argument("--adaptive", action="store_true",default=False, help="adaptive gap")
parser.add_argument("--out_weight", action="store_true",default=False, help="search for output nodes")
parser.add_argument("--init_alphas", type=float,default=0, help="initial weights of braches")
parser.add_argument("--reg_weight", type=float, default=2,help="operation regularizer")
parser.add_argument("--gap", type=float, default=100,help="gap for regularizer")
parser.add_argument("--sparsity", type=float, default=0,help="sparsity for operations")
parser.add_argument("--rand_label", action="store_true",default=False, help="use rand_labeld search")
parser.add_argument("--rand_data", action="store_true",default=False, help="use rand_data data")
parser.add_argument("--save", type=str, default="naspi",help="experiment name")
parser.add_argument("--seed", type=int, default=5, help="exp seed")
args = parser.parse_args()


args.save = "search/{}-{}-{}-{}-{}-{}".format(
    args.save,
    args.search_space,
    'ADA' if args.adaptive else 'FIX',
    'RL' if args.rand_label else 'L',
    'RD' if args.rand_data else 'D',
    f'REG{args.reg_weight}-G{args.gap}-S{args.seed}'
)
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = False
        cudnn.deterministic = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    if args.search_space == "normal":
        Network = NormalNetwork
    else:
        Network = ReduceNetwork

    torch.manual_seed(args.seed)

    model = Network(
        args.init_channels, # default 16
        CIFAR_CLASSES, # default 10
        args.layers, # default 9
        init_alphas=args.init_alphas, # default 0
        gumbel=args.gumbel, # default true
        out_weight=args.out_weight
    )

    if torch.cuda.is_available():
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    train_transform, _ = utils._data_transforms_cifar10(args)

    train_data = torchvision.datasets.CIFAR10(
        root=args.data, train=True, download=True, transform=train_transform,
    )

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )


    arch = NASISearch().score(arch=model, pre_defined=args, train_data=train_queue)
    logging.info("Final architecture: %s", (arch,))


