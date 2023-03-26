



import argparse


def sampler_args(parser):
    # EA sampler's parameters,
    parser.add_argument('--population_size', type=int, default=10, help="The learning rate for REINFORCE.")
    parser.add_argument('--sample_size', type=int, default=3, help="The momentum value for EMA.")


def space201_101_share_args(parser):
    parser.add_argument('--api_loc', type=str, default="NAS-Bench-201-v1_1-096897.pth",
                        help='which search space file to use, ['
                             'nasbench101: nasbench_only108.pkl'
                             'nasbench201: NAS-Bench-201-v1_1-096897.pth'
                             ' ... ]')

    parser.add_argument('--init_channels', default=10, type=int, help='output channels of stem convolution')


def nb101_args(parser):
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='# modules per stack')


def nb201_args(parser):
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--arch_size', type=int, default=3,
                        help='How many node the architecture has at least')


def mlp_args(parser):
    parser.add_argument('--num_layers', default=4, type=int, help='# hidden layers')
    parser.add_argument('--hidden_choice_len', default=20, type=int, help='number of hidden layer choices, 10 or 20')


def trainner_args(parser):
    parser.add_argument('--epoch', type=int, default=19,
                        help='number of maximum epochs, '
                             'frappe: 20, uci_diabetes: 20, criteo: 100'
                             'nb101: 108, nb201: 200')

    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=0.002, help="learning reate")
    parser.add_argument('--patience', type=int, default=1, help='number of epochs for stopping training')
    # parser.add_argument('--eval_freq', type=int, default=10000, help='max number of batches to train per epoch')

    parser.add_argument('--iter_per_epoch', type=int, default=200,
                        help="None, "
                             "200 for frappe, uci_diabetes, "
                             "2000 for criteo")

    # MLP model config
    parser.add_argument('--nfeat', type=int, default=5500,
                        help='the number of features, frappe: 5500, uci_diabetes: 369, criteo: 2100000')
    parser.add_argument('--nfield', type=int, default=10,
                        help='the number of fields, frappe: 10, uci_diabetes: 43, criteo: 39')
    parser.add_argument('--nemb', type=int, default=10,
                        help='embedding size')

    # MLP train config
    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')
    parser.add_argument('--workers', default=4, type=int, help='data loading workers')


def data_set_config(parser):
    parser.add_argument('--base_dir', type=str, default="/Users/kevin/project_python/firmest_data/",
                        help='path of data and result parent folder')
    # define search space,
    parser.add_argument('--dataset', type=str, default='frappe',
                        help='cifar10, cifar100, ImageNet16-120, '
                             'frappe, criteo, uci_diabetes')

    parser.add_argument('--num_labels', type=int, default=1,
                        help='[10, 100, 120],'
                             '[2, 2, 2]')


def seq_train_all_params(parser):
    parser.add_argument('--worker_id', type=int, default=0, help='start from 0')
    parser.add_argument('--total_workers', type=int, default=120, help='total number of workers')
    parser.add_argument('--total_models_per_worker', type=int, default=-1, help='How many models to evaluate')
    parser.add_argument('--pre_partitioned_file',
                        default="./exps/main_sigmod/ground_truth/sampled_models_10000_models.json",
                        type=str, help='num GPus')


def dis_train_all_models(parser):
    parser.add_argument('--worker_each_gpu', default=6, type=int, help='num worker each gpu')
    parser.add_argument('--gpu_num', default=8, type=int, help='num GPus')


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    # job config
    parser.add_argument('--log_name', type=str, default="main_T_100s", help="file name to store the log")
    parser.add_argument('--budget', type=int, default=600, help="Given budget, in second")

    # define search space,
    parser.add_argument('--search_space', type=str, default="mlp_sp",
                        help='search space [nasbench101, nasbench201, mlp_sp]')

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--device', type=str, default="cpu")

    parser.add_argument('--log_folder', default="LogCriteo", type=str, help='num GPus')

    sampler_args(parser)

    nb101_args(parser)
    nb201_args(parser)
    space201_101_share_args(parser)

    mlp_args(parser)
    data_set_config(parser)
    trainner_args(parser)
    seq_train_all_params(parser)
    dis_train_all_models(parser)

    return parser.parse_args()

