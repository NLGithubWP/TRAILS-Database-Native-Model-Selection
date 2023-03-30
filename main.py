# this is the main function of model selection.

import argparse
import calendar
import os
import time

from common.constant import Config
from storage.structure_data_loader import libsvm_dataloader


def sampler_args(parser):
    # EA sampler's parameters,
    parser.add_argument('--population_size', type=int, default=10, help="The learning rate for REINFORCE.")
    parser.add_argument('--sample_size', type=int, default=3, help="The momentum value for EMA.")


def space201_101_share_args(parser):
    parser.add_argument('--api_loc', type=str, default="NAS-Bench-201-v1_1-096897.pth",
                        help='which search space file to use, ['
                             'nasbench101: nasbench_only108.pkl'
                             'nasbench201: NAS-Bench-201-v1_1-096897.pth'
                             ' ... ]')

    parser.add_argument('--init_channels', default=10, type=int, help='output channels of stem convolution')


def nb101_args(parser):
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='# modules per stack')


def nb201_args(parser):
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--arch_size', type=int, default=3,
                        help='How many node the architecture has at least')


def mlp_args(parser):
    parser.add_argument('--num_layers', default=4, type=int, help='# hidden layers')
    parser.add_argument('--hidden_choice_len', default=20, type=int, help='number of hidden layer choices, 10 or 20')


def trainner_args(parser):
    parser.add_argument('--epoch', type=int, default=20,
                        help='number of maximum epochs, '
                             'frappe: 20, uci_diabetes: 20, criteo: 100'
                             'nb101: 108, nb201: 200')

    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=0.002, help="learning reate")
    parser.add_argument('--patience', type=int, default=1, help='number of epochs for stopping training')
    # parser.add_argument('--eval_freq', type=int, default=10000, help='max number of batches to train per epoch')

    parser.add_argument('--iter_per_epoch', type=int, default=200,
                        help="None, "
                             "200 for frappe, uci_diabetes, "
                             "2000 for criteo")

    # MLP model config
    parser.add_argument('--nfeat', type=int, default=5500, help='the number of features')
    parser.add_argument('--nfield', type=int, default=10, help='the number of fields')
    parser.add_argument('--nemb', type=int, default=10, help='embedding size')

    # MLP train config
    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')
    parser.add_argument('--workers', default=4, type=int, help='data loading workers')


def data_set_config(parser):
    parser.add_argument('--base_dir', type=str, default="/Users/kevin/project_python/firmest_data/",
                        help='path of data and result parent folder')
    # define search space,
    parser.add_argument('--dataset', type=str, default='frappe',
                        help='cifar10, cifar100, ImageNet16-120, '
                             'frappe, criteo, uci_diabetes')

    parser.add_argument('--num_labels', type=int, default=1,
                        help='[10, 100, 120],'
                             '[2, 2, 2]')


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--log_name', type=str, default="main_T_100s", help="file name to store the log")
    parser.add_argument('--budget', type=int, default=600, help="Given budget, in second")

    # define search space,
    parser.add_argument('--search_space', type=str, default="mlp_sp",
                        help='search space [nasbench101, nasbench201, mlp_sp]')

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--device', type=str, default="cpu")

    sampler_args(parser)

    nb101_args(parser)
    nb201_args(parser)
    space201_101_share_args(parser)

    mlp_args(parser)
    data_set_config(parser)
    trainner_args(parser)

    return parser.parse_args()


def generate_data_loader():
    if args.dataset in [Config.c10, Config.c100, Config.imgNet]:
        train_loader, val_loader, class_num = dataset.get_dataloader(
            train_batch_size=args.batch_size,
            test_batch_size=args.batch_size,
            dataset=args.dataset,
            num_workers=1,
            datadir=os.path.join(args.base_dir, "data"))
        test_loader = val_loader
    else:
        train_loader, val_loader, test_loader = libsvm_dataloader(
            args=args,
            data_dir=os.path.join(args.base_dir, "data", "structure_data", args.dataset),
            nfield=args.nfield,
            batch_size=args.batch_size)
        class_num = args.num_labels

    return train_loader, val_loader, test_loader, class_num


def run_with_time_budget(time_budget: float):
    """
    :param time_budget: the given time budget, in second
    :return:
    """

    # define dataLoader, and sample a mini-batch
    train_loader, val_loader, test_loader, class_num = generate_data_loader()
    args.num_labels = class_num
    data_loader = [train_loader, val_loader, test_loader]

    rms = RunModelSelection(args.search_space,  args, is_simulate=True)
    best_arch, _, _, _ = rms.select_model_online(time_budget, data_loader)

    return best_arch


if __name__ == "__main__":
    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from eva_engine.run_ms import RunModelSelection
    from storage import dataset

    run_with_time_budget(args.budget)
