# this is the main function of model selection.

import calendar
import os
import time
from distutils.util import strtobool
from exps.shared_args import parse_arguments
from src.tools.compute import log_scale_x_array


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


def run_with_time_budget(time_budget: float, is_simulate: bool, only_phase1: bool):
    """
    :param time_budget: the given time budget, in second
    :return:
    """

    # define dataLoader, and sample a mini-batch

    data_loader = [train_loader, val_loader, test_loader]

    rms = RunModelSelection(args.search_space, args, is_simulate=is_simulate)
    best_arch, best_arch_performance, time_usage, _, _, p1_trace_highest_score, p1_trace_models_perforamnces = \
        rms.select_model_online(
            budget=time_budget,
            data_loader=data_loader,
            only_phase1=only_phase1,
            run_workers=1)

    return best_arch, best_arch_performance, time_usage, p1_trace_highest_score, p1_trace_models_perforamnces


if __name__ == "__main__":
    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.eva_engine.run_ms import RunModelSelection
    from src.storage import dataset
    from src.common.constant import Config
    from src.logger import logger
    from src.storage.structure_data_loader import libsvm_dataloader
    from src.tools.io_tools import write_json

    train_loader, val_loader, test_loader, class_num = generate_data_loader()
    args.num_labels = class_num

    # configurable settings for benchmarking
    is_simulate = args.is_simulate
    only_phase1 = args.only_phase1
    # for this exp, we repeat 100 times and set max to 1000 mins
    total_run = 50
    max_minute = 1000
    budget_array = log_scale_x_array(num_points=args.num_points, max_minute=max_minute)

    if only_phase1:
        checkpoint_name = f"./internal/ml/model_selection/exp_result/" \
                          f"res_end_2_end_{args.search_space}_{args.dataset}_{args.kn_rate}_{args.num_points}_p1.json"
    else:
        checkpoint_name = f"./internal/ml/model_selection/exp_result/" \
                          f"res_end_2_end_{args.search_space}_{args.dataset}_{args.kn_rate}_{args.num_points}.json"

    print(checkpoint_name)
    result = {
        "sys_time_budget": budget_array,
        "sys_acc": []
    }
    for run_id in range(total_run):
        run_begin_time = time.time()
        run_acc_list = []
        for time_budget in budget_array:
            time_budget_sec = time_budget * 60
            logger.info(f"\n Running job with budget={time_budget} min \n")
            best_arch, best_arch_performance, time_usage, p1_trace_highest_score, p1_trace_models_perforamnces = \
                run_with_time_budget(time_budget_sec,
                                     is_simulate=is_simulate,
                                     only_phase1=only_phase1)

            run_acc_list.append(best_arch_performance)
        result["sys_acc"].append(run_acc_list)

        print(f"finish run_id = {run_id}, using {time.time() - run_begin_time}")

        # checkpointing each run
        write_json(checkpoint_name, result)
