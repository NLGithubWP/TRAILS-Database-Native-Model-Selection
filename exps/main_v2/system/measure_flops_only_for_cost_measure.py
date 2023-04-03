

import argparse
import calendar
import json
import os
import time
import traceback
import sys

from exps.main_v2.common.shared_args import parse_arguments


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
    from search_space.init_search_space import init_search_space
    from storage.structure_data_loader import libsvm_dataloader

    search_space_ins = init_search_space(args)
    search_space_ins.load()

    try:
        # read the checkpoint
        checkpoint_file_name = f"./{args.dataset}_train_cfg_exp/train_config_tune_{args.dataset}_epo_{args.epoch}.json"

        # 1. data loader
        train_loader, val_loader, test_loader = libsvm_dataloader(
            args=args,
            data_dir=os.path.join(args.base_dir, "data", "structure_data", args.dataset),
            nfield=args.nfield,
            batch_size=args.batch_size)

        arch_id = "256-256-256-256"
        print(f"begin to train the {arch_id}")

        from thop import clever_format, profile
        model = search_space_ins.new_architecture(arch_id).to(args.device)
        for batch_idx, batch in enumerate(train_loader):
            target = batch['y'].to(args.device)
            batch['id'] = batch['id'].to(args.device)
            batch['value'] = batch['value'].to(args.device)

            flops, params = profile(model, inputs=(batch,))
            backward_flops = flops
            total_flops_for_one_epoch = (flops + backward_flops) * args.iter_per_epoch
            flops2, params2 = clever_format([flops, params], "%.3f")
            print('Train FLOPs = ' + str(total_flops_for_one_epoch / 1000 ** 3) + 'G')
            print('Train Params = ' + str(params / 1000 ** 2) + 'M')
            # print("origin: ", f"arch_id={3380}, B={args.batch_size},C={args.init_channels},
            # flops2={flops2}， params2={params2}")
            break

        for batch_idx, batch in enumerate(val_loader):
            target = batch['y'].to(args.device)
            batch['id'] = batch['id'].to(args.device)
            batch['value'] = batch['value'].to(args.device)

            flops, params = profile(model, inputs=(batch,))
            backward_flops = flops
            total_flops_for_one_epoch = flops * len(val_loader) / args.batch_size
            flops2, params2 = clever_format([flops, params], "%.3f")
            print('Val FLOPs = ' + str(total_flops_for_one_epoch / 1000 ** 3) + 'G')
            print('Val Params = ' + str(params / 1000 ** 2) + 'M')
            # print("origin: ", f"arch_id={3380}, B={args.batch_size},C={args.init_channels},
            # flops2={flops2}， params2={params2}")
            break

    except:
        print(traceback.format_exc())

