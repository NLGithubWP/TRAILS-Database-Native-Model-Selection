

import argparse
import calendar
import json
import os
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--log_name', type=str, default="baseline_train_based", help="file name to store the log")

    # define search space,
    parser.add_argument('--search_space', type=str, default="mlp_sp", help='[nasbench101, nasbench201, mlp_sp]')
    parser.add_argument('--num_layers', default=4, type=int, help='# hidden layers')
    parser.add_argument('--hidden_choice_len', default=10, type=int, help='number of hidden layer choices, 10 or 20')

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default="/Users/kevin/project_python/firmest_data/",
                        help='path of data and result parent folder')

    # define search space,
    parser.add_argument('--num_labels', type=int, default=1, help='[10, 100, 120, 2, 2, 2]')

    # those are for training
    parser.add_argument('--device', type=str, default="cpu")

    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")

    parser.add_argument('--epoch', type=int, default=1,
                        help='number of maximum epochs, frappe: 20, uci_diabetes: 20, criteo: 100')
    parser.add_argument('--iter_per_epoch', type=int, default=3,
                        help="None, or some number, Iteration per epoch, it is controlled by scheduler")

    parser.add_argument('--dataset', type=str, default='criteo',
                        help='cifar10, cifar100, ImageNet16-120, frappe, criteo, uci_diabetes')
    # MLP model config
    parser.add_argument('--nfeat', type=int, default=2100000,
                        help='the number of features, frappe: 5500, uci_diabetes: 369, criteo: 2100000')
    parser.add_argument('--nfield', type=int, default=39,
                        help='the number of fields, frappe: 10, uci_diabetes: 43, criteo: 39')
    parser.add_argument('--nemb', type=int, default=10,
                        help='embedding size')

    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')

    parser.add_argument('--worker_id', type=int, default=0, help='start from 0')
    parser.add_argument('--total_workers', type=int, default=120, help='total number of workers')
    parser.add_argument('--total_models_per_worker', type=int, default=-1, help='How many models to evaluate')

    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')

    parser.add_argument('--log_folder', default="LogCriteo", type=str, help='num GPus')

    parser.add_argument('--pre_partitioned_file',
                        default="./exps/main_sigmod/ground_truth/sampled_models_10000_models.json",
                        type=str, help='num GPus')

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
    os.environ.setdefault("log_file_name", f"{args.log_name}_{args.dataset}_wkid_{args.worker_id}_{ts}.log")
    os.environ.setdefault("base_dir", args.base_dir)

    from logger import logger
    from eva_engine.phase2.algo.trainer import ModelTrainer
    from search_space.init_search_space import init_search_space
    from storage.structure_data_loader import libsvm_dataloader
    from utilslibs.io_tools import write_json, read_json

    search_space_ins = init_search_space(args)
    search_space_ins.load()

    # 1. data loader
    logger.info(f" Loading data....")
    train_loader, val_loader, test_loader = libsvm_dataloader(
        args=args,
        data_dir=os.path.join(args.base_dir, "data", "structure_data", args.dataset),
        nfield=args.nfield,
        batch_size=args.batch_size)

    res = read_json(args.pre_partitioned_file)

    all_partition = partition_list_by_worker_id(list(res.keys()), args.total_workers)

    if args.total_models_per_worker == -1:
        logger.info(
            f" ---- begin exploring, current worker have  "
            f"{len(all_partition[args.worker_id])} models. explore all those models ")
    else:
        logger.info(f" ---- begin exploring, current worker have  "
                    f"{len(all_partition[args.worker_id])} models. but explore {args.total_models_per_worker} models ")

    # read the checkpoint
    checkpoint_file_name = f"./base_line_res_{args.dataset}/train_baseline_{args.dataset}_wkid_{args.worker_id}.json"
    visited = read_json(checkpoint_file_name)
    if visited == {}:
        visited = {args.dataset: {}}
        logger.info(f" ---- initialize checkpointing with {visited} . ")
    else:
        logger.info(f" ---- recovery from checkpointing with {len(visited[args.dataset])} model. ")

    explored_arch_num = 0
    for arch_index in all_partition[args.worker_id]:
        print(f"begin to train the {arch_index}")
        if res[arch_index] in visited[args.dataset]:
            logger.info(f" ---- model {res[arch_index]} already visited")
            continue
        valid_auc, _, train_log = ModelTrainer.fully_train_arch(
            search_space_ins=search_space_ins,
            arch_id=res[arch_index],
            use_test_acc=False,
            epoch_num=args.epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args)

        # update the shared model eval res
        logger.info(f" ---- exploring {explored_arch_num} model. ")
        logger.info(f" ---- info: {json.dumps({res[arch_index]:train_log})}")
        visited[args.dataset][res[arch_index]] = train_log
        explored_arch_num += 1
        
        if args.total_models_per_worker != -1 and explored_arch_num > args.total_models_per_worker:
            break

        logger.info(f" Saving result to: {checkpoint_file_name}")
        write_json(checkpoint_file_name, visited)
