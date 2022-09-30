import argparse
import os
import random
import time
import traceback
global args
import calendar
from common.structure import ModelEvaData, ModelAcquireData


def load_101_cfg(model_acquire: ModelAcquireData, bn: bool):
    matrix, operations = NasBench101Space.deserialize_model_encoding(model_acquire.model_encoding)

    model_cfg = NasBench101Cfg(
        bn=bn,
        init_channels=16,
        num_stacks=3,
        num_modules_per_stack=3,
        num_labels=args.num_labels
    )
    spec = ModelSpec(matrix, operations)
    model = NasBench101Network(spec, model_cfg)
    return model


def load_201_cfg(model_acquire: ModelAcquireData, bn: bool):
    model_cfg = NasBench201Cfg(
        bn=bn,
        init_channels=16,
        init_b_type="none",
        init_w_type="none",
        num_labels=args.num_labels
    )
    arch_hash = model_acquire.model_encoding

    architecture = nasbench2.get_model_from_arch_str(
        arch_hash,
        model_cfg.num_labels,
        model_cfg.bn,
        model_cfg.init_channels)

    init_net(architecture, model_cfg.init_w_type, model_cfg.init_b_type)
    return architecture


def load_dataset(args):
    # define dataLoader, and sample a mini-batch
    train_loader, val_loader, class_num = dataset.get_dataloader(
        train_batch_size=1,
        test_batch_size=1,
        dataset=args.dataset,
        num_workers=1,
        datadir=args.base_dir)
    args.num_labels = class_num

    # [random, grasp]
    mini_batch, mini_batch_targets = dataset.get_mini_batch(
        dataloader=train_loader,
        sample_alg="random",
        batch_size=32,
        num_classes=class_num)

    mini_batch = mini_batch.to(args.device)
    mini_batch_targets = mini_batch_targets.to(args.device)

    return mini_batch, mini_batch_targets


def mode_evaluation():
    """
    Evaluate the model in loop
    :param args:
    :return:
    """

    # pre_load data
    mini_batch, mini_batch_targets = load_dataset(args)
    # record time
    model_gene_t = []
    model_score_t = []
    total_t = []
    # pre_load model evaluation result
    model_eva = ModelEvaData()

    cfg_load_method = None
    if args.search_space == Config.NB101:
        cfg_load_method = load_101_cfg
    elif args.search_space == Config.NB201:
        cfg_load_method = load_201_cfg

    eval_num = 0

    while True:
        eval_num += 1
        # ask for new model
        data = model_eva.serialize_model()
        response = send_get_request(search_strategy_url, data)
        model_acquire = ModelAcquireData.deserialize(response.text)
        if model_acquire.is_last:
            logger.info("Worker receive is_last message and exit the execution")
            write_json(args.worker_save_file,
                       {"model_gene_t": model_gene_t,
                        "model_score_t": model_score_t,
                        "total_t": total_t
                        })
            break

        model_eva_begin = time.time()
        try:
            # score NasWot
            newmodel_load_begin_time = time.time()
            new_model = cfg_load_method(model_acquire, bn=True)
            new_model = new_model.to(args.device)
            newmodel_load_end_time = time.time()

            nw_begin_time = time.time()
            naswot_score, _ = evaluator_register[CommonVars.NAS_WOT].evaluate_wrapper(
                arch=new_model,
                device=args.device,
                batch_data=mini_batch,
                batch_labels=mini_batch_targets)
            nw_end_time = time.time()

            # score SynFlow
            synflow_load_begin_time = time.time()
            new_model = cfg_load_method(model_acquire, bn=False)
            new_model = new_model.to(args.device)
            synflow_load_end_time = time.time()
            model_gene_t.append(synflow_load_end_time - synflow_load_begin_time +
                                newmodel_load_end_time - newmodel_load_begin_time)

            synflow_begin_time = time.time()
            synflow_score, _ = evaluator_register[CommonVars.PRUNE_SYNFLOW].evaluate_wrapper(
                arch=new_model,
                device=args.device,
                batch_data=mini_batch,
                batch_labels=mini_batch_targets)
            synflow_end_time = time.time()
            model_score_t.append(synflow_end_time - synflow_begin_time + nw_end_time - nw_begin_time)

            model_score = {CommonVars.NAS_WOT: naswot_score,
                           CommonVars.PRUNE_SYNFLOW: synflow_score}

            model_eva.model_id = model_acquire.model_id
            model_eva.model_score = model_score
            total_t.append(time.time() - model_eva_begin)
        except:
            print(f"Model {model_acquire.model_id} has error")
            print(traceback.format_exc())
            logger.info(traceback.format_exc())


def parse_arguments():
    base_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    parser.add_argument('--log_name', type=str, default="wk1.log")
    parser.add_argument('--worker_save_file', type=str, default="wk_time_usage")

    parser.add_argument('--controller_url', type=str, default="http://0.0.0.0:8002")

    # job config define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--base_dir', type=str, default=os.path.join(base_dir, "data"), help='path of data folder')

    # dataLoader setting,
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--num_labels', type=int, default=10, help='[10, 100, 120]')

    # define search space,
    parser.add_argument('--search_space', type=str, default="nasbench101", help='[nasbench101, nasbench201 ...]')

    # define device
    parser.add_argument('--device', type=str, default="cpu", help='which device to use [cpu, cuda]')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_file_name", args.log_name)
    random.seed(10)
    from apiserver.client import send_get_request
    from logger import logger
    from search_space.utils.weight_initializers import init_net
    from storage import dataset
    from common.constant import CommonVars
    from eva_engine import evaluator_register
    from search_space import NasBench101Cfg, NasBench101Space, NasBench201Cfg
    from search_space.nas_101_api.lib.model import NasBench101Network
    from search_space.nas_101_api.lib.nb101_api import ModelSpec
    from search_space.nas_201_api.lib import nasbench2
    from common.constant import Config
    from utilslibs.tools import write_json

    search_strategy_url = args.controller_url + "/get_model"
    mode_evaluation()

