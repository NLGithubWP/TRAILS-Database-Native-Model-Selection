

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

    from search_space.init_search_space import init_search_space
    from storage import dataset
    from common.constant import Config
    from storage.structure_data_loader import libsvm_dataloader
    from utilslibs.io_tools import write_json

    # train_loader, val_loader, test_loader, class_num = generate_data_loader()
    # args.num_labels = class_num

    checkpoint_name = f"./exps/main_v2/analysis/result/" \
                      f"res_end_2_end_{args.dataset}_{args.kn_rate}_{args.num_points}_db4nas.json"

    print(f"begin to explore {args.db4nas_n} models, and keep {args.db4nas_k} models, "
          f"with data = {args.batch_data}")

    import json
    the_data = json.loads(args.batch_data)

    print(len(the_data))
    print(the_data)
    print(list(the_data[0].values()))

    # run_acc_list = []
    # search_space_ins = init_search_space(args)
    # data_loader = [train_loader, val_loader, test_loader]
    #
    # # run phase-1 to get the K models.
    # p1_runner = RunPhase1(
    #     args=args,
    #     K=args.db4nas_k, N=args.db4nas_n,
    #     search_space_ins=search_space_ins,
    #     train_loader=train_loader,
    #     is_simulate=True)
    #
    #
    # result = {
    #     "models": K_models,
    # }
    #
    # # checkpointing each run
    # write_json(checkpoint_name, result)





















