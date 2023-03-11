
import argparse
import os
import random
from eva_engine.run_ms import RunModelSelection
from utilslibs.draw_tools import get_plot_compare_with_base_line_cfg, get_base_annotations
from utilslibs.parse_pre_res import SimulateTrain
from query_api.query_train_baseline_api import post_processing_train_base_result
from utilslibs.io_tools import write_json, read_json


def default_args(parser):
    parser.add_argument('--init_channels', default=16, type=int, help='output channels of stem convolution')
    # search space configs for nasBench101
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='# modules per stack')

    #  weight initialization settings, nasBench201 requires it.
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')

    parser.add_argument('--bn', type=int, default=1,
                        help="If use batch norm in network 1 = true, 0 = false")

    # nas101 doesn't need this, while 201 need it.
    parser.add_argument('--arch_size', type=int, default=1,
                        help='How many node the architecture has at least')

    # EA sampler's parameters,
    parser.add_argument('--population_size', type=int, default=10, help="The learning rate for REINFORCE.")
    parser.add_argument('--sample_size', type=int, default=3, help="The momentum value for EMA.")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default=os.path.join(base_dir, "data"),
                        help='path of data folder')


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--controller_port', type=int, default=8002)
    parser.add_argument('--log_name', type=str, default="SS_1wk_1run_NB101_c10.log")

    parser.add_argument('--run', type=int, default=100, help="how many run")
    parser.add_argument('--save_file_latency', type=str,
                        default=os.path.join(base_dir, "1wk_1run_NB101_c10_latency"), help="search target")
    parser.add_argument('--save_file_all', type=str,
                        default=os.path.join(base_dir, "1wk_1run_NB101_c10_all"), help="search target")

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench101",
                        help='search space to use, [nasbench101, nasbench201, ... ]')

    # define search space,
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--num_labels', type=int, default=10, help='[10, 100, 120]')

    parser.add_argument('--api_loc', type=str, default="nasbench_only108.pkl",
                        help='which search space to use, ['
                             'nasbench101: nasbench_only108.pkl'
                             'nasbench201: NAS-Bench-201-v1_1-096897.pth'
                             ' ... ]')

    default_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    base_dir = os.getcwd()
    random.seed(10)
    args = parse_arguments()

    # this is for acquire the final acc
    fgt = SimulateTrain(space_name=args.search_space, total_epoch=200)

    saved_dict = read_json(f"./0_macro_res_{args.search_space}_{args.dataset}")

    # phase1 + phase2
    run_range_, budget_array, sub_graph_y1, sub_graph_y2, sub_graph_split, draw_graph = \
        get_plot_compare_with_base_line_cfg(args.search_space, args.dataset, False)

    y_acc_list_arr = []
    for run_id in run_range_:
        y_each_run = []
        for Tmin in budget_array:
            best_arch, _, _, _ = \
                RunModelSelection(args.search_space, args.dataset, is_simulate=True).\
                    select_model_simulate(Tmin*60, run_id, only_phase1=False)
            acc_sh_v, _ = fgt.get_ground_truth(arch_id=best_arch, dataset=args.dataset, epoch_num=None)
            y_each_run.append(acc_sh_v)
        y_acc_list_arr.append(y_each_run)

    # phase1
    run_range_, budget_array_p1, sub_graph_y1, sub_graph_y2, sub_graph_split, draw_graph = \
        get_plot_compare_with_base_line_cfg(args.search_space, args.dataset, True)

    y_acc_list_arr_p1 = []
    for run_id in run_range_:
        y_each_run_p1 = []
        for Tmin in budget_array_p1:
            # phase1
            best_arch_p1, _, _, _ = \
                RunModelSelection(args.search_space, args.dataset, is_simulate=True).\
                    select_model_simulate(Tmin * 60, only_phase1=True)
            # 4. Training it and getting the real accuracy.rain and get the final acc
            acc_sh_v_p1, _ = fgt.get_ground_truth(arch_id=best_arch_p1, dataset=args.dataset, epoch_num=None)
            y_each_run_p1.append(acc_sh_v_p1)
        y_acc_list_arr_p1.append(y_each_run_p1)

    x_acc_train, y_acc_train_l, y_acc_train_m, y_acc_train_h = post_processing_train_base_result(
        search_space=args.search_space, dataset=args.dataset)

    saved_dict["y_acc_list_arr"] = y_acc_list_arr
    saved_dict["x_T_list"] = budget_array
    saved_dict["x_acc_train"] = x_acc_train
    saved_dict["y_acc_train_l"] = y_acc_train_l
    saved_dict["y_acc_train_m"] = y_acc_train_m
    saved_dict["y_acc_train_h"] = y_acc_train_h
    saved_dict["y_acc_list_arr_only_phase1"] = y_acc_list_arr_p1
    saved_dict["x_T_list_only_phase1"] = budget_array_p1

    write_json(f"./0_macro_res_{args.search_space}_{args.dataset}", saved_dict)

    draw_graph(y_acc_list_arr, budget_array, y_acc_list_arr_p1, budget_array_p1,
            x_acc_train, y_acc_train_l, y_acc_train_m, y_acc_train_h,
            get_base_annotations(args.dataset), sub_graph_split,
            f"{args.search_space}_{args.dataset}", args.dataset,
            sub_graph_y1,
            sub_graph_y2)

