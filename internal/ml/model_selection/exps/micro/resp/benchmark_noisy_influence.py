import calendar
import os
import time
from exps.shared_args import parse_arguments
from src.tools.compute import log_scale_x_array
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or any other supported backend
from matplotlib import pyplot as plt
import numpy as np


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


def select_with_noise(all_explored_models, search_space_ins, K, noise_degree=0.1):
    # Number of models to be selected from top K and from the rest
    top_K_count = int(K * (1 - noise_degree))
    rest_count = K - top_K_count

    # Selecting 90% of the top K models
    top_K_selected = random.sample(all_explored_models[-K:], top_K_count)

    # Selecting 10% from the rest of the models
    rest_selected = []
    for _ in range(rest_count):
        while True:
            arch_id, arch_micro = search_space_ins.random_architecture_id()
            if arch_id not in rest_selected:
                rest_selected.append(arch_id)
                break

    # Combining the selected models
    selected_models = top_K_selected + rest_selected
    print(f" --- sample {len(top_K_selected)} from top K, {len(rest_selected)} from rest")
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
    print(f"Total explored {len(all_models)} models, select top {K} models.")

    degree_auc = []
    for noise_degree in noisy_degree:
        k_models = select_with_noise(all_models, rms.search_space_ins, K, noise_degree)
        best_arch, best_arch_performance, _ = rms.refinement_phase(U, k_models)
        print(f"noise_degree={noise_degree}, best_arch_performance={best_arch_performance}")
        degree_auc.append(best_arch_performance)
    return degree_auc


def plot_experiment(exp_list, title):
    def plot_exp(time_usg, exp, label):
        exp = np.array(exp)
        q_75_y = np.quantile(exp, .75, axis=0)
        q_25_y = np.quantile(exp, .25, axis=0)
        mean_y = np.mean(exp, axis=0)

        print(
            f"T = {time_budget * 60} sec, noisy_degree ={noisy_degree}, "
            f"sys_acc_m={mean_y}, sys_acc_m_25={q_25_y}")

        plt.plot(time_usg, mean_y, "-*", label=label, )
        plt.fill_between(time_usg, q_25_y, q_75_y, alpha=0.1)

    fig, ax = plt.subplots()

    for time_usg, exp, ename in exp_list:
        plot_exp(time_usg, exp, ename)
    plt.grid()
    plt.xlabel('Time in mins')
    plt.ylabel('Test Accuracy')

    plt.legend()
    plt.title(title)
    fig.savefig(f"./internal/ml/model_selection/exp_result/noisy_inflence.pdf",
                bbox_inches='tight')


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

    total_run = 5
    budget_array = [1, 10]

    noisy_degree = [0, 0.1, 0.2, 0.5, 1]
    all_lines_auc = {}
    for ele in noisy_degree:
        all_lines_auc[f"noisy degree - {ele}"] = []

    for run_id in range(total_run):
        # here each run have one list
        for ele in noisy_degree:
            all_lines_auc[f"noisy degree - {ele}"].append([])

        for time_budget in budget_array:
            run_begin_time = time.time()
            time_budget_sec = time_budget * 60
            logger.info(f"\n Running job with budget={time_budget} min \n")
            # _degree_auc: [AUC_noisy1, AUC_noisy2...]
            _degree_auc = run_with_time_budget(
                time_budget_sec,
                is_simulate=is_simulate,
                only_phase1=only_phase1)

            for idx in range(len(noisy_degree)):
                ele = noisy_degree[idx]
                all_lines_auc[f"noisy degree - {ele}"][run_id].append(_degree_auc[idx])

    # draw the graph
    draw_list = []
    for key, value in all_lines_auc.items():
        one_line = (
            budget_array, value, key
        )
        draw_list.append(one_line)
    plot_experiment(draw_list, "Noisy Degree")
