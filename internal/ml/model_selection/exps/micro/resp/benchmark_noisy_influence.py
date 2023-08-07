import calendar
import os
import time
from exps.shared_args import parse_arguments
from src.tools.compute import log_scale_x_array
import random


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


def select_with_noise(k_models, K, noise_degree=0.1):
    # Number of models to be selected from top K and from the rest
    top_K_count = int(K * (1 - noise_degree))
    rest_count = K - top_K_count

    # Selecting 90% of the top K models
    top_K_selected = random.sample(k_models[-K:], top_K_count)

    # Selecting 10% from the rest of the models
    rest_selected = random.sample(k_models[:-K], rest_count)

    # Combining the selected models
    selected_models = top_K_selected + rest_selected

    return selected_models


def run_with_time_budget(time_budget: float, is_simulate: bool, only_phase1: bool):
    """
    :param time_budget: the given time budget, in second
    :return:
    """

    # define dataLoader, and sample a mini-batch

    rms = RunModelSelection(args.search_space, args, is_simulate=is_simulate)

    score_time_per_model = rms.profile_filtering()
    train_time_per_epoch = rms.profile_refinement()
    K, U, N = rms.coordination(time_budget, score_time_per_model, train_time_per_epoch, only_phase1)
    k_models, all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id = rms.filtering_phase(N, K)

    for noise_degree in [0, 0.1, 0.2, 0.5, 1]:
        k_models = select_with_noise(all_models, K, noise_degree)
        best_arch, best_arch_performance, _ = rms.refinement_phase(U, k_models)
        print(f"noise_degree={noise_degree}, best_arch_performance={best_arch_performance}")

    return best_arch_performance


if __name__ == "__main__":
    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.eva_engine.run_ms import RunModelSelection
    from src.dataset_utils import dataset
    from src.common.constant import Config
    from src.logger import logger
    from src.dataset_utils.structure_data_loader import libsvm_dataloader
    from src.tools.io_tools import write_json

    train_loader, val_loader, test_loader, class_num = generate_data_loader()
    args.num_labels = class_num

    # configurable settings for benchmarking
    is_simulate = True
    only_phase1 = False

    total_run = 1
    budget_array = [100]

    for run_id in range(total_run):
        run_begin_time = time.time()
        run_acc_list = []
        for time_budget in budget_array:
            time_budget_sec = time_budget * 60
            logger.info(f"\n Running job with budget={time_budget} min \n")
            run_with_time_budget(time_budget_sec,
                                 is_simulate=is_simulate,
                                 only_phase1=only_phase1)

        print(f"finish run_id = {run_id}, using {time.time() - run_begin_time}")
