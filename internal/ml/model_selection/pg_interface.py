import calendar
import os
import time

import json
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import traceback
import orjson
import argparse
import configparser
from argparse import Namespace


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

    return args


def exception_catcher(func):
    def wrapper(encoded_str: str):
        try:
            # each functon accepts a json string
            params = json.loads(encoded_str)
            config_file = params.pop("config_file")

            # Parse the config file
            args = parse_config_arguments(config_file)

            # Set the environment variables
            ts = calendar.timegm(time.gmtime())
            os.environ.setdefault("base_dir", args.base_dir)
            os.environ.setdefault("log_logger_folder_name", args.log_folder)
            os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")

            # Call the original function with the parsed parameters
            return func(params, args)
        except Exception as e:
            return orjson.dumps(
                {"Errored": traceback.format_exc()}).decode('utf-8')

    return wrapper


class LibsvmDataset(Dataset):
    """ Dataset loader for Libsvm data format """

    @staticmethod
    def decode_libsvm(columns):
        map_func = lambda pair: (int(pair[0]), float(pair[1]))
        id, value = zip(*map(lambda col: map_func(col.split(':')), columns[:-1]))
        sample = {'id': torch.LongTensor(id),
                  'value': torch.FloatTensor(value),
                  'y': float(columns[-1])}
        return sample

    @staticmethod
    def pre_processing(mini_batch_data: List[Dict]):
        sample_lines = len(mini_batch_data)
        nfields = len(mini_batch_data[0].keys()) - 1
        feat_id = torch.LongTensor(sample_lines, nfields)
        feat_value = torch.FloatTensor(sample_lines, nfields)
        y = torch.FloatTensor(sample_lines)

        for i in range(sample_lines):
            row_value = mini_batch_data[i].values()
            sample = LibsvmDataset.decode_libsvm(list(row_value))
            feat_id[i] = sample['id']
            feat_value[i] = sample['value']
            y[i] = sample['y']
        return feat_id, feat_value, y, sample_lines

    def __init__(self, mini_batch_data: List[Dict]):
        self.feat_id, self.feat_value, self.y, self.nsamples = \
            LibsvmDataset.pre_processing(mini_batch_data)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return {'id': self.feat_id[idx],
                'value': self.feat_value[idx],
                'y': self.y[idx]}


@exception_catcher
def model_selection(params: dict, args: Namespace):
    # define dataLoader, and sample a mini-batch

    mini_batch_data = json.loads(params["mini_batch"])
    budget = float(params["budget"])

    from src.logger import logger
    from src.eva_engine.run_ms import RunModelSelection

    logger.info(f"begin run filtering phase at {os.getcwd()}")

    dataloader = DataLoader(LibsvmDataset(mini_batch_data),
                            batch_size=args.batch_size,
                            shuffle=True)

    data_loader = [dataloader, dataloader, dataloader]

    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    best_arch, best_arch_performance, time_usage, _, _, _, p1_trace_highest_score, p1_trace_highest_scored_models_id = \
        rms.select_model_online(
            budget=budget,
            data_loader=data_loader,
            only_phase1=False,
            run_workers=1)

    return orjson.dumps(
        {"best_arch": best_arch,
         "best_arch_performance": best_arch_performance,
         "time_usage": time_usage}).decode('utf-8')


@exception_catcher
def profiling_filtering_phase(params: dict, args: Namespace):
    # define dataLoader, and sample a mini-batch

    mini_batch_m = params["mini_batch"]

    from src.logger import logger
    from src.eva_engine.run_ms import RunModelSelection

    logger.info(f"begin run filtering phase at {os.getcwd()}, with {mini_batch_m}")

    mini_batch_data = json.loads(mini_batch_m)
    dataloader = DataLoader(LibsvmDataset(mini_batch_data),
                            batch_size=args.batch_size,
                            shuffle=True)
    data_loader = [dataloader, dataloader, dataloader]

    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    score_time_per_model = rms.profile_filtering(data_loader=data_loader)

    return orjson.dumps({"time": score_time_per_model}).decode('utf-8')


@exception_catcher
def profiling_refinement_phase(params: dict, args: Namespace):
    # define dataLoader, and sample a mini-batch

    mini_batch_m = params["mini_batch"]

    from src.logger import logger
    from src.eva_engine.run_ms import RunModelSelection

    logger.info(f"begin run refinement phase at {os.getcwd()}")
    mini_batch_data = json.loads(mini_batch_m)

    dataloader = DataLoader(LibsvmDataset(mini_batch_data),
                            batch_size=args.batch_size,
                            shuffle=True)
    data_loader = [dataloader, dataloader, dataloader]

    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    train_time_per_epoch = rms.profile_refinement(data_loader=data_loader)

    # mini_batch_data = mini_batch.decode('utf-8')
    return orjson.dumps({"time": train_time_per_epoch}).decode('utf-8')


@exception_catcher
def coordinator(params: dict, args: Namespace):
    budget = float(params["budget"])
    score_time_per_model = float(params["score_time_per_model"])
    train_time_per_epoch = float(params["train_time_per_epoch"])
    only_phase1 = True if params["only_phase1"].lower() == "true" else False

    from src.logger import logger
    from src.eva_engine.run_ms import RunModelSelection

    logger.info(f"coordinator params: budget={budget}, "
                f"score_time_per_model={score_time_per_model}, "
                f"train_time_per_epoch={train_time_per_epoch}, "
                f"only_phase1={only_phase1}")

    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    K, U, N = rms.coordination(
        budget=budget,
        score_time_per_model=score_time_per_model,
        train_time_per_epoch=train_time_per_epoch,
        only_phase1=only_phase1)

    return orjson.dumps(
        {"k": K, "u": U, "n": N}).decode('utf-8')


@exception_catcher
def filtering_phase(params: dict, args: Namespace):
    mini_batch_m = params["mini_batch"]
    n = int(params["n"])
    k = int(params["k"])

    from src.logger import logger
    from src.eva_engine.run_ms import RunModelSelection

    logger.info(f"begin run filtering phase at {os.getcwd()}")

    mini_batch_data = json.loads(mini_batch_m)
    dataloader = DataLoader(LibsvmDataset(mini_batch_data),
                            batch_size=args.batch_size,
                            shuffle=True)

    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    k_models = rms.filtering_phase(N=n, K=k, train_loader=dataloader)

    return orjson.dumps({"k_models": k_models}).decode('utf-8')


@exception_catcher
def refinement_phase(params: dict, args: Namespace):
    mini_batch_m = params["mini_batch"]
    return orjson.dumps(
        {"k_models": "k_models"}).decode('utf-8')


@exception_catcher
def model_selection_workloads(params: dict, args: Namespace):
    """
    Run filtering (explore N models) and refinement phase (refine K models) for benchmarking latency.
    """
    mini_batch_m = params["mini_batch"]
    n = int(params["n"])
    k = int(params["k"])

    from src.logger import logger
    from src.eva_engine.run_ms import RunModelSelection
    logger.info(f"begin run model_selection_workloads ")
    mini_batch_data = json.loads(mini_batch_m)
    dataloader = DataLoader(LibsvmDataset(mini_batch_data),
                            batch_size=args.batch_size,
                            shuffle=True)
    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    k_models = rms.filtering_phase(N=n, K=k, train_loader=dataloader)
    best_arch, best_arch_performance = rms.refinement_phase(
        U=1,
        k_models=k_models,
        train_loader=dataloader,
        valid_loader=dataloader)

    return orjson.dumps(
        {"best_arch": best_arch,
         "best_arch_performance": best_arch_performance,
         }).decode('utf-8')


@exception_catcher
def test_io(params: dict, args: Namespace):
    return orjson.dumps({"inputs are": json.dumps(params)}).decode('utf-8')


if __name__ == "__main__":
    params = {}
    params["budget"] = 10
    params["score_time_per_model"] = 0.0211558125
    params["train_time_per_epoch"] = 5.122203075885773
    params["only_phase1"] = 'true'
    params["config_file"] = './internal/ml/model_selection/config.ini'
    print(coordinator(json.dumps(params)))

    params = {}
    params[
        "mini_batch"] = '[{"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}]'
    params["n"] = 10
    params["k"] = 1
    params["config_file"] = './internal/ml/model_selection/config.ini'
    print(filtering_phase(json.dumps(params)))

    params = {}
    params[
        "mini_batch"] = '[{"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}]'
    params["config_file"] = './internal/ml/model_selection/config.ini'
    print(profiling_refinement_phase(json.dumps(params)))

    params = {}
    params["budget"] = 10
    params[
        "mini_batch"] = '[{"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}]'
    params["config_file"] = './internal/ml/model_selection/config.ini'
    print(model_selection(json.dumps(params)))
