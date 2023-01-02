import argparse
import os
import random
import time
from sanic import Sanic
from sanic.response import json
import calendar

app = Sanic("FastAutoNAS")
global used_search_space
global gt_api
global args
global rmngr


@app.route('/select_model', methods=['GET'])
async def select_model(request):
    clientReq = ClientStruct.deserialize(request.json)
    budget = clientReq.budget
    dataset = clientReq.dataset


@app.route('/')
async def hello(request):
    return json({'hello': 'world'})


@app.route('/')
async def hello(request):
    return json({'hello': 'world'})


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

    # default don't run tps experiment
    parser.add_argument('--throughput_run_models', type=int, default=0)
    # job config
    parser.add_argument('--controller_port', type=int, default=8002)
    parser.add_argument('--log_name', type=str, default="SS_1wk_1run_NB101_c10.log")

    parser.add_argument('--run', type=int, default=1, help="how many run")
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
                             'nasbench201: NAS-Bench-201-v1_0-e61699.pth'
                             'nasbench201: NAS-Bench-201-v1_1-096897.pth'
                             ' ... ]')

    default_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    base_dir = os.getcwd()
    random.seed(10)

    args = parse_arguments()
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_file_name", args.log_name)

    from common.constant import Config
    from common.structure import ModelAcquireData, ModelEvaData, ClientStruct
    from query_api.gt_api import Gt201, Gt101
    from controller.controler import Controller
    import search_space
    from search_space import NasBench101Space
    from utilslibs.tools import write_json
    from logger import logger

    if args.search_space == Config.NB201:
        gt_api = Gt201()
    elif args.search_space == Config.NB101:
        gt_api = Gt101()
    used_search_space = search_space.init_search_space(args)

    logger.info("start server")
    app.run(host="0.0.0.0", port=args.controller_port, debug=False, access_log=False)
