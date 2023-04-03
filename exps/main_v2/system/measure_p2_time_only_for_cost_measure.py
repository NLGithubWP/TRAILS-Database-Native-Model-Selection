

# this is the main function of model selection.

import calendar
import os
import time
from exps.main_v2.common.shared_args import parse_arguments


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


if __name__ == "__main__":
    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from eva_engine.phase2.run_sh import BudgetAwareControllerSH

    from eva_engine.phase1.run_phase1 import RunPhase1
    from search_space.init_search_space import init_search_space
    from storage import dataset
    from common.constant import Config
    from storage.structure_data_loader import libsvm_dataloader

    train_loader, val_loader, test_loader, class_num = generate_data_loader()
    args.num_labels = class_num

    print(f"begin to explore {args.db4nas_n} models, and keep {args.db4nas_k} models, with data = {args.batch_data}")

    run_acc_list = []
    search_space_ins = init_search_space(args)
    data_loader = [train_loader, val_loader, test_loader]

    # 0. profiling dataset and search space, get t1 and t2
    score_time_per_model, train_time_per_epoch, N_K_ratio = search_space_ins.profiling(
        args.dataset,
        train_loader,
        val_loader,
        args,
        is_simulate=True)

    sh = BudgetAwareControllerSH(
        search_space_ins=search_space_ins,
        dataset_name=args.dataset,
        eta=3,
        time_per_epoch=train_time_per_epoch,
        is_simulate=True,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args)

    _, total_time = sh.pre_calculate_time_required(647, 1)
    print(total_time)






