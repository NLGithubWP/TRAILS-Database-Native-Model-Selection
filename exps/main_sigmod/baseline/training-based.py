import torch

from common.structure import ModelEvaData
import argparse
import calendar
import os
import time


def default_args(parser):

    # EA sampler's parameters,
    parser.add_argument('--population_size', type=int, default=10, help="The learning rate for REINFORCE.")
    parser.add_argument('--sample_size', type=int, default=3, help="The momentum value for EMA.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--log_name', type=str, default="baseline_train_based", help="file name to store the log")
    parser.add_argument('--budget', type=int, default=300, help="Given budget, in second")

    # define search space,
    parser.add_argument('--search_space', type=str, default="mlp_sp",
                        help='search space [nasbench101, nasbench201, mlp_sp]')

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="/Users/kevin/project_python/firmest_data/",
                        help='path of data and result parent folder')

    # define search space,
    parser.add_argument('--dataset', type=str, default='frappe',
                        help='cifar10, cifar100, ImageNet16-120, '
                             'frappe, movielens, uci_diabetes')

    parser.add_argument('--num_labels', type=int, default=1,
                        help='[10, 100, 120],'
                             '[2, 2, 2]')

    # those are for training
    parser.add_argument('--device', type=str, default="cpu")

    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=0.002, help="learning reate")
    parser.add_argument('--patience', type=int, default=1, help='number of epochs for stopping training')
    # parser.add_argument('--eval_freq', type=int, default=1, help='max number of batches to train per epoch')

    parser.add_argument('--N', type=int, default=5000, help='How many arch to train')

    parser.add_argument('--epoch', type=int, default=2, help='number of maximum epochs')
    parser.add_argument('--iter_per_epoch', type=int, default=None,
                        help="None, or some number, Iteration per epoch, it is controlled by scheduler")

    # MLP model config
    parser.add_argument('--num_layers', default=4, type=int, help='# hidden layers')
    parser.add_argument('--nfeat', type=int, default=5500, help='the number of features')
    parser.add_argument('--nfield', type=int, default=10, help='the number of fields')
    parser.add_argument('--nemb', type=int, default=10, help='embedding size')

    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')

    default_args(parser)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_file_name", args.log_name+"_"+str(ts)+".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from controller import RegularizedEASampler
    from controller.controler import SampleController
    from eva_engine.phase2.algo.trainer import ModelTrainer
    from search_space.init_search_space import init_search_space
    from storage.structure_data_loader import libsvm_dataloader

    search_space_ins = init_search_space(args)
    search_space_ins.load()

    # 1. seq: init the search strategy and controller,
    strategy = RegularizedEASampler(search_space_ins,
                                    population_size=args.population_size,
                                    sample_size=args.sample_size)

    sampler = SampleController(strategy)
    arch_generator = sampler.sample_next_arch()

    # 1. data loader
    train_loader, val_loader, test_loader = libsvm_dataloader(
        data_dir=os.path.join(args.base_dir, "data", "structure_data", args.dataset),
        nfield=args.nfield,
        batch_size=args.batch_size,
        workers=1)

    explored_n = 0
    model_eva = ModelEvaData()

    base_line_log = {args.dataset: {}}
    while explored_n < args.N:
        if explored_n > 0:
            sampler.fit_sampler(model_eva.model_id, model_eva.model_score, use_prue_score=True)

        explored_n += 1
        arch_id, arch_micro = arch_generator.__next__()
        f1score, _, train_log = ModelTrainer.fully_train_arch(
               search_space_ins=search_space_ins,
               arch_id=arch_id,
               use_test_acc=True,
               epoch_num=args.epoch,
               train_loader=train_loader,
               val_loader=val_loader,
               test_loader=test_loader,
               args=args)

        # update the shared model eval res
        model_eva.model_id = str(arch_id)
        model_eva.model_score = {"f1score": f1score}

        base_line_log[args.dataset][arch_id] = train_log





