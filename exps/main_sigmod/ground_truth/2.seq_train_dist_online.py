

import argparse
import calendar
import json
import logging
import os
import time
import torch.multiprocessing as mp


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

    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")

    parser.add_argument('--epoch', type=int, default=20,
                        help='number of maximum epochs, frappe: 20, uci_diabetes: 20, criteo: 100')
    parser.add_argument('--iter_per_epoch', type=int, default=200,
                        help="None, or some number, Iteration per epoch, it is controlled by scheduler")

    parser.add_argument('--dataset', type=str, default='frappe',
                        help='cifar10, cifar100, ImageNet16-120, frappe, criteo, uci_diabetes')
    # MLP model config
    parser.add_argument('--nfeat', type=int, default=5500,
                        help='the number of features, frappe: 5500, uci_diabetes: 369, criteo: 2100000')
    parser.add_argument('--nfield', type=int, default=10,
                        help='the number of fields, frappe: 10, uci_diabetes: 43, criteo: 39')
    parser.add_argument('--nemb', type=int, default=10,
                        help='embedding size')

    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')

    parser.add_argument('--total_models_per_worker', type=int, default=None, help='How many models to evaluate')

    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')

    parser.add_argument('--worker_each_gpu', default=6, type=int, help='num worker each gpu')
    parser.add_argument('--gpu_num', default=8, type=int, help='num GPus')

    parser.add_argument('--log_folder', default="Logs", type=str, help='num GPus')

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


def start_one_worker(queue, args, worker_id, my_partition, search_space_ins, res):
    from utilslibs.io_tools import write_json, read_json
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    os.environ.setdefault("log_file_name", f"{args.log_name}_{args.dataset}_wkid_{worker_id}_{ts}.log")
    # import logging
    logger = logging.getLogger(f"{args.dataset}_wkid_{worker_id}_{ts}")
    if not os.path.exists(f"./{args.log_folder}"):
        os.makedirs(f"./{args.log_folder}")
    handler = logging.FileHandler(f"./{args.log_folder}/{args.log_name}_{args.dataset}_wkid_{worker_id}_{ts}.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    from eva_engine.phase2.algo.trainer import ModelTrainer

    if args.total_models_per_worker is None:
        logger.info(
            f" ---- begin exploring, current worker have  "
            f"{len(my_partition)} models. explore all those models ")
    else:
        logger.info(f" ---- begin exploring, current worker have  "
                    f"{len(my_partition)} models. but explore {args.total_models_per_worker} models ")

    train_loader, val_loader, test_loader = queue.get()

    checkpoint_file_name = f"./base_line_res_{args.dataset}/train_baseline_{args.dataset}_wkid_{worker_id}.json"
    visited = read_json(checkpoint_file_name)
    if visited == {}:
        visited = {args.dataset: {}}
        logger.info(f" ---- initialize checkpointing with {visited} . ")
    else:
        logger.info(f" ---- recovery from checkpointing with {len(visited[args.dataset])} model. ")

    explored_arch_num = 0
    for arch_index in my_partition:
        print(f"begin to train the {arch_index}")
        valid_auc, _, train_log = ModelTrainer.fully_train_arch(
            search_space_ins=search_space_ins,
            arch_id=res[arch_index],
            use_test_acc=False,
            epoch_num=args.epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args, logger=logger)

        # update the shared model eval res
        logger.info(f" ---- exploring {explored_arch_num} model. ")
        logger.info(f" ---- info: {json.dumps({res[arch_index]: train_log})}")
        visited[args.dataset][res[arch_index]] = train_log
        explored_arch_num += 1

        if args.total_models_per_worker is not None and explored_arch_num > args.total_models_per_worker:
            break

        logger.info(f" Saving result to: {checkpoint_file_name}")
        write_json(checkpoint_file_name, visited)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    os.environ.setdefault("log_file_name", f"{args.log_name}_{args.dataset}_main_{ts}.log")
    os.environ.setdefault("base_dir", args.base_dir)

    from search_space.init_search_space import init_search_space
    from storage.structure_data_loader import libsvm_dataloader
    from utilslibs.io_tools import write_json, read_json

    search_space_ins = init_search_space(args)
    search_space_ins.load()

    # 1. main process partition data and group results,
    res = read_json(args.pre_partitioned_file)

    total_workers = args.worker_each_gpu * args.gpu_num
    all_partition = partition_list_by_worker_id(list(res.keys()), total_workers)

    train_loader, val_loader, test_loader = libsvm_dataloader(
        args=args,
        data_dir=os.path.join(args.base_dir, "data", "structure_data", args.dataset),
        nfield=args.nfield,
        batch_size=args.batch_size)

    # 2. put the shared dataloader into the queue,
    queue = mp.Queue()

    # 3. Create a list of processes to train the models
    processes = []
    worker_id = 0
    for gpu_id in range(args.gpu_num):
        for _ in range(args.worker_each_gpu):
            if args.device != "cpu":
                args.device = f"cuda:{gpu_id}"
            print(f"running process {[args.device, worker_id, len(all_partition[worker_id])]}")
            p = mp.Process(
                target=start_one_worker,
                args=(queue, args, worker_id, all_partition[worker_id], search_space_ins, res,
                      )
            )
            p.start()
            processes.append(p)
            worker_id += 1

    # 4. send to the queue
    for gpu_id in range(args.gpu_num):
        for _ in range(args.worker_each_gpu):
            print("putting to queue ....")
            queue.put((train_loader, val_loader, test_loader))

    print("All processing are running, waiting all to finish....")
    for p in processes:
        p.join()


