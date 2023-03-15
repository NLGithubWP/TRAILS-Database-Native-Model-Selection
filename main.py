# this is the main function of model selection.

import argparse
import calendar
import os
import time


def default_args(parser):
    parser.add_argument('--init_channels', default=16, type=int, help='output channels of stem convolution')
    # search space configs for nasBench101
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='# modules per stack')

    #  weight initialization settings, nasBench201 requires it.
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')

    parser.add_argument('--bn', type=int, default=1,
                        help="If use batch norm in network 1 = true, 0 = false")

    # nas101 doesn't need this, while 201 need it.
    parser.add_argument('--arch_size', type=int, default=3,
                        help='How many node the architecture has at least')

    # EA sampler's parameters,
    parser.add_argument('--population_size', type=int, default=10, help="The learning rate for REINFORCE.")
    parser.add_argument('--sample_size', type=int, default=3, help="The momentum value for EMA.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--log_name', type=str, default="main_T_100s", help="file name to store the log")
    parser.add_argument('--budget', type=int, default=250, help="Given budget, in second")

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench201",
                        help='search space [nasbench101, nasbench201, ... ]')

    # define search space,
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--num_labels', type=int, default=10, help='[10, 100, 120]')

    parser.add_argument('--api_loc', type=str, default="NAS-Bench-201-v1_1-096897.pth",
                        help='which search space file to use, ['
                             'nasbench101: nasbench_only108.pkl'
                             'nasbench201: NAS-Bench-201-v1_1-096897.pth'
                             ' ... ]')

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="/Users/kevin/project_python/firmest_data/",
                        help='path of data and result parent folder')

    default_args(parser)
    return parser.parse_args()


def run_with_time_budget(time_budget: float):
    """
    :param time_budget: the given time budget, in second
    :return:
    """

    # define dataLoader, and sample a mini-batch
    train_loader, val_loader, class_num = dataset.get_dataloader(
        train_batch_size=1,
        test_batch_size=1,
        dataset=args.dataset,
        num_workers=1,
        datadir=os.path.join(args.base_dir, "data"))
    args.num_labels = class_num

    rms = RunModelSelection(args.search_space, args.dataset, is_simulate=False)
    best_arch, _, _, _ = rms.select_model_online(time_budget, train_loader, args)

    return best_arch


if __name__ == "__main__":

    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_file_name", args.log_name+"_"+str(ts)+".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from eva_engine.run_ms import RunModelSelection
    from storage import dataset

    run_with_time_budget(args.budget)
