import calendar
import os
import time
import argparse
import configparser
from sanic import Sanic
from sanic.exceptions import InvalidUsage
from sanic.response import json

ts = calendar.timegm(time.gmtime())
os.environ.setdefault("log_logger_folder_name", "log_eval_service")
os.environ.setdefault("log_file_name", "eval_service_" + str(ts) + ".log")
from src.logger import logger
from src.eva_engine.run_ms import RunModelSelection
from src.dataset_utils.stream_dataloader import StreamingDataLoader


def parse_config_arguments(config_path: str):
    parser = configparser.ConfigParser()
    parser.read(config_path)

    args = argparse.Namespace()

    # job config under DEFAULT
    args.log_name = parser.get('DEFAULT', 'log_name')
    args.budget = parser.getint('DEFAULT', 'budget')
    args.device = parser.get('DEFAULT', 'device')
    args.log_folder = parser.get('DEFAULT', 'log_folder')
    args.result_dir = parser.get('DEFAULT', 'result_dir')
    args.num_points = parser.getint('DEFAULT', 'num_points')
    args.max_load = parser.getint('DEFAULT', 'max_load')

    # sampler args
    args.search_space = parser.get('SAMPLER', 'search_space')
    args.population_size = parser.getint('SAMPLER', 'population_size')
    args.sample_size = parser.getint('SAMPLER', 'sample_size')
    args.simple_score_sum = parser.getboolean('SAMPLER', 'simple_score_sum')

    # nb101 args
    args.api_loc = parser.get('NB101', 'api_loc')
    args.init_channels = parser.getint('NB101', 'init_channels')
    args.bn = parser.getint('NB101', 'bn')
    args.num_stacks = parser.getint('NB101', 'num_stacks')
    args.num_modules_per_stack = parser.getint('NB101', 'num_modules_per_stack')

    # nb201 args
    args.init_w_type = parser.get('NB201', 'init_w_type')
    args.init_b_type = parser.get('NB201', 'init_b_type')
    args.arch_size = parser.getint('NB201', 'arch_size')

    # mlp args
    args.num_layers = parser.getint('MLP', 'num_layers')
    args.hidden_choice_len = parser.getint('MLP', 'hidden_choice_len')

    # mlp_trainer args
    args.epoch = parser.getint('MLP_TRAINER', 'epoch')
    args.batch_size = parser.getint('MLP_TRAINER', 'batch_size')
    args.lr = parser.getfloat('MLP_TRAINER', 'lr')
    args.patience = parser.getint('MLP_TRAINER', 'patience')
    args.iter_per_epoch = parser.getint('MLP_TRAINER', 'iter_per_epoch')
    args.nfeat = parser.getint('MLP_TRAINER', 'nfeat')
    args.nfield = parser.getint('MLP_TRAINER', 'nfield')
    args.nemb = parser.getint('MLP_TRAINER', 'nemb')
    args.report_freq = parser.getint('MLP_TRAINER', 'report_freq')
    args.workers = parser.getint('MLP_TRAINER', 'workers')

    # dataset args
    args.base_dir = parser.get('DATASET', 'base_dir')
    args.dataset = parser.get('DATASET', 'dataset')
    args.num_labels = parser.getint('DATASET', 'num_labels')

    # seq_train args
    args.worker_id = parser.getint('SEQ_TRAIN', 'worker_id')
    args.total_workers = parser.getint('SEQ_TRAIN', 'total_workers')
    args.total_models_per_worker = parser.getint('SEQ_TRAIN', 'total_models_per_worker')
    args.pre_partitioned_file = parser.get('SEQ_TRAIN', 'pre_partitioned_file')

    # dis_train args
    args.worker_each_gpu = parser.getint('DIS_TRAIN', 'worker_each_gpu')
    args.gpu_num = parser.getint('DIS_TRAIN', 'gpu_num')

    # tune_interval args
    args.kn_rate = parser.getint('TUNE_INTERVAL', 'kn_rate')

    # anytime args
    args.only_phase1 = parser.getboolean('ANYTIME', 'only_phase1')
    args.is_simulate = parser.getboolean('ANYTIME', 'is_simulate')

    args.refinement_url = parser.get('SERVER', 'refinement_url')
    args.cache_svc_url = parser.get('SERVER', 'cache_svc_url')

    args.db_name = parser.get('DB_CONFIG', 'db_name')
    args.db_user = parser.get('DB_CONFIG', 'db_user')
    args.db_host = parser.get('DB_CONFIG', 'db_host')
    args.db_port = parser.get('DB_CONFIG', 'db_port')

    return args


def refinement_phase(u, k_models, table_name, config_file):
    args = parse_config_arguments(config_file)

    train_dataloader = StreamingDataLoader(cache_svc_url=args.cache_svc_url, table_name=table_name, name_space="train")
    eval_dataloader = StreamingDataLoader(cache_svc_url=args.cache_svc_url, table_name=table_name, name_space="valid")

    try:
        rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
        best_arch, best_arch_performance, _ = rms.refinement_phase(
            U=u,
            k_models=k_models,
            train_loader=train_dataloader,
            valid_loader=eval_dataloader)
    finally:
        train_dataloader.stop()
        eval_dataloader.stop()
    return {"best_arch": best_arch, "best_arch_performance": best_arch_performance}


app = Sanic("CacheServiceApp")


@app.route("/", methods=["POST"])
async def start_refinement_phase(request):
    # Check if request is JSON
    if not request.json:
        logger.info("Expecting JSON payload")
        raise InvalidUsage("Expecting JSON payload")

    u = request.json.get('u')
    k_models = request.json.get('k_models')
    table_name = request.json.get('table_name')
    config_file = request.json.get('config_file')

    if u is None or k_models is None or config_file is None:
        logger.info(f"Missing 'u' or 'k_models' in JSON payload, {request.json}")
        raise InvalidUsage("Missing 'u' or 'k_models' in JSON payload")

    result = refinement_phase(u, k_models, table_name, config_file)

    return json(result)


if __name__ == "__main__":
    import requests
    url = 'http://localhost:8093/'
    columns = ['col1', 'col2', 'col3', 'label']
    response = requests.post(
      url, json={'columns': columns,
                 'name_space': "train",
                 'table_name': "dummy",
                 "batch_size": 32})
    print(response.json())

    response = requests.post(
      url, json={'columns': columns,
                 'name_space': "valid",
                 'table_name': "dummy",
                 "batch_size": 32})
    print(response.json())

    # this is filtering phase
    time.sleep(5)

    result = refinement_phase(1, ["8-8-8-8", "16-16-16-16"],
                              "dummy",
                              "/project/TRAILS/internal/ml/model_selection/config.ini")

    # app.run(host="0.0.0.0", port=8095)
