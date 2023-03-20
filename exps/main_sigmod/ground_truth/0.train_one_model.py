

import argparse
import calendar
import json
import os
import time
import traceback
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--log_name', type=str, default="baseline_train_based", help="file name to store the log")

    # define search space,
    parser.add_argument('--search_space', type=str, default="mlp_sp", help='[nasbench101, nasbench201, mlp_sp]')
    parser.add_argument('--num_layers', default=4, type=int, help='# hidden layers')

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="/Users/kevin/project_python/firmest_data/",
                        help='path of data and result parent folder')

    # define search space,
    parser.add_argument('--num_labels', type=int, default=1, help='[10, 100, 120, 2, 2, 2]')

    # those are for training
    parser.add_argument('--device', type=str, default="cpu")

    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")

    parser.add_argument('--epoch', type=int, default=1,
                        help='number of maximum epochs, frappe: 20, uci_diabetes: 20, criteo: 100')
    parser.add_argument('--iter_per_epoch', type=int, default=None,
                        help="None, or some number, Iteration per epoch, it is controlled by scheduler")

    parser.add_argument('--dataset', type=str, default='frappe',
                        help='cifar10, cifar100, ImageNet16-120, frappe, criteo, uci_diabetes')
    # MLP model config
    parser.add_argument('--nfeat', type=int, default=2100000,
                        help='the number of features, frappe: 5500, uci_diabetes: 369, criteo: 2100000')
    parser.add_argument('--nfield', type=int, default=39,
                        help='the number of fields, frappe: 10, uci_diabetes: 43, criteo: 39')
    parser.add_argument('--nemb', type=int, default=10,
                        help='embedding size')

    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')

    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')

    parser.add_argument('--log_folder', default="LogCriteo", type=str, help='num GPus')

    return parser.parse_args()


def partition_list_by_worker_id(lst, num_workers=15):
    partitions = []
    for i in range(num_workers):
        partitions.append([])
    for idx, item in enumerate(lst):
        worker_id = idx % num_workers
        partitions[worker_id].append(item)
    return partitions


if __name__ == "__main__":

    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", f"{args.log_name}_{args.dataset}_ep{args.epoch}_{ts}.log")
    os.environ.setdefault("base_dir", args.base_dir)

    from logger import logger
    from eva_engine.phase2.algo.trainer import ModelTrainer
    from search_space.init_search_space import init_search_space
    from storage.structure_data_loader import libsvm_dataloader
    from utilslibs.io_tools import write_json, read_json

    search_space_ins = init_search_space(args)
    search_space_ins.load()

    try:
        # read the checkpoint
        checkpoint_file_name = f"./criteo_train_cfg_exp/train_config_tune_{args.dataset}_epo_{args.epoch}.json"

        # 1. data loader
        train_loader, val_loader, test_loader = libsvm_dataloader(
            args=args,
            data_dir=os.path.join(args.base_dir, "data", "structure_data", args.dataset),
            nfield=args.nfield,
            batch_size=args.batch_size)

        arch_id = "256-256-256-256"
        print(f"begin to train the {arch_id}")

        valid_auc, _, train_log = ModelTrainer.fully_train_arch(
            search_space_ins=search_space_ins,
            arch_id=arch_id,
            use_test_acc=False,
            epoch_num=args.epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args)

        # update the shared model eval res
        logger.info(f" ---- info: {json.dumps({arch_id:train_log})}")

        logger.info(f" Saving result to: {checkpoint_file_name}")
        write_json(checkpoint_file_name, train_log)
    except:
        logger.info(traceback.format_exc())

