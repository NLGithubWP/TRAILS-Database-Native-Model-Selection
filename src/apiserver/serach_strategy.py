import argparse
import os
import random
import time
import traceback

from sanic import Sanic
from sanic.response import text, json
import calendar


app = Sanic("FastAutoNAS")
global used_search_space
global gt_api
global args
global rmngr


class RunStatus:
    def __init__(self, run_id: int):

        self.run_id = run_id
        self.max_explore_model = 5000

        self.sampler = Controller(used_search_space, args)
        self.arch_generator = self.sampler.sample_next_arch(args.arch_size)

        # begin time
        self.last_finish_time = None
        self.x_time_current = 0

        self.x_axis_time = []
        self.y_axis_top10_models = []
        self.model_is_ls = []

        # {target_key: latency}
        self.run_res = {}

    def fit_sampler(self, model_eva):
        if model_eva.model_id is not None:
            self.sampler.fit_sampler(
                model_eva.model_id,
                model_eva.model_score)

    def get_next_model(self):
        arch_id, model_struc = self.arch_generator.__next__()
        if used_search_space.name == Config.NB101:
            model_encoding = NasBench101Space.serialize_model_encoding(
                model_struc.original_matrix.tolist(),
                model_struc.original_ops)
            test_accuracy, time_usage = gt_api.get_c10_test_info(arch_id=arch_id)
        elif used_search_space.name == Config.NB201:
            model_encoding = used_search_space.archid_to_hash(arch_id)
            test_accuracy, time_usage = gt_api.query_200_epoch(arch_id=str(arch_id), dataset=args.dataset)
        else:
            model_encoding, test_accuracy = None, None

        return arch_id, model_encoding, test_accuracy

    def record_model_eval_res(self, arch_id):
        # if this is the first time the controller start a new run
        if self.last_finish_time is None:
            self.last_finish_time = time.time()
            return

        self.model_is_ls.append(arch_id)
        # record time used in evaluation
        time_used = time.time() - self.last_finish_time
        self.x_time_current += time_used
        self.x_axis_time.append(self.x_time_current)
        # update current finish time.
        self.last_finish_time = time.time()

        self.y_axis_top10_models.append(self.sampler.get_current_top_k_models())
        if len(self.x_axis_time) % 50 == 0:
            print(f"run {self.run_id} evaluated models num {len(self.model_is_ls)}")
            logger.info(f"run {self.run_id} evaluated models num {len(self.model_is_ls)}")
        self.update_run_res()

    def update_run_res(self):
        # query the last one
        i = -1
        current_time = self.x_axis_time[i]
        current_top_10 = self.y_axis_top10_models[i]
        high_acc = get_high_acc_top_10(current_top_10)

        if len(self.x_axis_time) >= 2:
            for key, value in target.items():
                if key in self.run_res:
                    continue
                if high_acc > target[key]:
                    # self.x_axis_time[0] is the first time used to serve the init request
                    self.run_res[key] = current_time

    def is_run_finish(self):
        # if set throughput_run_models, then we run throughput_run_models exp
        if args.throughput_run_models > 0:
            return len(self.model_is_ls) > args.throughput_run_models
        else:
            # if find all targets or explore more than 5k models. stop exploring
            if len(target) == len(self.run_res) or len(self.model_is_ls) > self.max_explore_model:
                return True
            else:
                return False

    def fetch_run_satus(self):
        return \
            {"arch_is_ls": self.model_is_ls,
             "x_axis_time": self.x_axis_time,
             "y_axis_top10_models": self.y_axis_top10_models}


class RunManager:
    def __init__(self):
        # status
        self.current_run = None
        # {run_id: {...}}
        self.global_res = {}
        self.all_run_info = {}

    def switch_new_run(self):

        if self.current_run is None:
            logger.info("Switch a new run with id 1")
            self.current_run = RunStatus(1)
        elif self.current_run.is_run_finish():
            # fetch above result
            self.global_res[self.current_run.run_id] = self.current_run.run_res
            self.all_run_info[self.current_run.run_id] = self.current_run.fetch_run_satus()
            if self.current_run.run_id == args.run:
                write_json(args.save_file_latency, self.global_res)
                write_json(args.save_file_all, self.all_run_info)
            new_run_id = self.current_run.run_id + 1
            self.current_run = RunStatus(new_run_id)
            logger.info(f"Switch a new run with id {new_run_id}")
        return self.current_run


def get_target_latency():
    if used_search_space.name == Config.NB101:
        c10target = {"938%": 0.938, "937%": 0.937, "935%": 0.935, "93%": 0.93}
        # c10target = {"test%": 0}
        return c10target
    if used_search_space.name == Config.NB201:
        c10target = {"938%": 0.938, "937%": 0.937, "935%": 0.935, "93%": 0.93}
        return c10target


def get_ground_truth(arch_id):
    if used_search_space.name == Config.NB101:
        score_, _ = gt_api.get_c10_test_info(arch_id)
        return score_
    if used_search_space.name == Config.NB201:
        score_, _ = gt_api.query_200_epoch(arch_id, args.dataset)
        return score_


# get the high acc of k arch with highest score
def get_high_acc_top_10(top10):
    if len(top10) == 0:
        return 0
    all_top10_acc = []
    for arch_id in top10:
        score_ = get_ground_truth(arch_id)
        all_top10_acc.append(score_)
    return max(all_top10_acc)


@app.route('/get_model', methods=['GET'])
async def explore_models(request):

    try:
        # communication, receive
        model_eva = ModelEvaData.deserialize(request.json)

        rs = rmngr.switch_new_run()

        # fit sampler, None means first time acquire model
        rs.fit_sampler(model_eva)
        # generate new model
        arch_id, model_encoding, test_accuracy = rs.get_next_model()
        # record arch_id with higher score
        rs.record_model_eval_res(arch_id)

        # communication, send
        model_acquire_data = ModelAcquireData(model_id=str(arch_id), model_encoding=model_encoding)
        if rs.run_id >= args.run + 1:
            model_acquire_data.is_last = True
            logger.info(f"Finish this job, please kill the {args.controller_port}")
        data_str = model_acquire_data.serialize_model()
        return text(data_str)
    except:
        logger.info(traceback.format_exc())


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
    from common.structure import ModelAcquireData, ModelEvaData
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

    rmngr = RunManager()
    target = get_target_latency()

    logger.info("start server")
    app.run(host="0.0.0.0", port=args.controller_port, debug=False, access_log=False)
