import calendar
import os
import time
from exps.shared_args import parse_arguments

import json
from typing import List, Dict
import torch
from torch.utils.data import Dataset
import traceback
import orjson
from src.tools.io_tools import write_json

def exception_catcher(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            write_json(
                "/project/TRAILS/log_score_time_frappe/test.log",
                {"Errored": traceback.format_exc()})
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
        nfields = 3
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
def model_selection(mini_batch_m: str):
    # define dataLoader, and sample a mini-batch

    args = parse_arguments()
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    args.log_name = "score_based_all_metrics"
    args.search_space = "mlp_sp"
    args.num_labels = 2
    args.device = "cpu"
    args.batch_size = 8
    args.dataset = "frappe"
    args.base_dir = "/project/exp_data/"
    args.result_dir = "/project/TRAILS/internal/ml/model_selection/exp_result/"

    os.environ.setdefault("log_logger_folder_name", "/project/TRAILS/log_score_time_frappe")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.logger import logger
    from src.eva_engine.run_ms import RunModelSelection

    logger.info(f"begin run filtering phase at {os.getcwd()}")
    mini_batch_data = json.loads(mini_batch_m)

    dataloader = LibsvmDataset(mini_batch_data)
    data_loader = [dataloader, dataloader, dataloader]

    rms = RunModelSelection(args.search_space, args, is_simulate=True)
    best_arch, best_arch_performance, time_usage, _, _, _, p1_trace_highest_score, p1_trace_highest_scored_models_id = \
        rms.select_model_online(
            budget=2,
            data_loader=data_loader,
            only_phase1=True,
            run_workers=1)

    # mini_batch_data = mini_batch.decode('utf-8')
    return orjson.dumps(
        {"best_arch": best_arch,
         "best_arch_performance": best_arch_performance,
         "time_usage": time_usage,
         "p1_trace_highest_score": p1_trace_highest_score,
         "p1_trace_highest_scored_models_id": p1_trace_highest_scored_models_id,
         }).decode('utf-8')


@exception_catcher
def profiling_filtering_phase(mini_batch_m: str):
    # define dataLoader, and sample a mini-batch

    args = parse_arguments()
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    args.log_name = "score_based_all_metrics"
    args.search_space = "mlp_sp"
    args.num_labels = 2
    args.device = "cpu"
    args.batch_size = 8
    args.dataset = "frappe"
    args.base_dir = "/project/exp_data/"
    args.result_dir = "/project/TRAILS/internal/ml/model_selection/exp_result/"

    os.environ.setdefault("log_logger_folder_name", "/project/TRAILS/log_score_time_frappe")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.logger import logger
    from src.eva_engine.run_ms import RunModelSelection

    logger.info(f"begin run filtering phase at {os.getcwd()}, with {mini_batch_m}")
    mini_batch_data = json.loads(mini_batch_m)

    dataloader = LibsvmDataset(mini_batch_data)
    data_loader = [dataloader, dataloader, dataloader]

    rms = RunModelSelection(args.search_space, args, is_simulate=True)
    score_time_per_model = rms.profile_filtering(data_loader=data_loader)

    # mini_batch_data = mini_batch.decode('utf-8')
    return orjson.dumps(
        {"time": score_time_per_model}).decode('utf-8')


@exception_catcher
def profiling_refinement_phase(mini_batch_m: str):
    # define dataLoader, and sample a mini-batch

    args = parse_arguments()
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    args.log_name = "score_based_all_metrics"
    args.search_space = "mlp_sp"
    args.num_labels = 2
    args.device = "cpu"
    args.batch_size = 8
    args.dataset = "frappe"
    args.base_dir = "/project/exp_data/"
    args.result_dir = "/project/TRAILS/internal/ml/model_selection/exp_result/"

    os.environ.setdefault("log_logger_folder_name", "/project/TRAILS/log_score_time_frappe")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.logger import logger
    from src.eva_engine.run_ms import RunModelSelection

    logger.info(f"begin run refinement phase at {os.getcwd()}")
    mini_batch_data = json.loads(mini_batch_m)

    dataloader = LibsvmDataset(mini_batch_data)
    data_loader = [dataloader, dataloader, dataloader]

    rms = RunModelSelection(args.search_space, args, is_simulate=True)
    train_time_per_epoch = rms.profile_refinement(data_loader=data_loader)

    # mini_batch_data = mini_batch.decode('utf-8')
    return orjson.dumps(
        {"time": train_time_per_epoch}).decode('utf-8')


@exception_catcher
def coordinator(encode_str: str):
    # define dataLoader, and sample a mini-batch
    args = parse_arguments()
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    args.log_name = "score_based_all_metrics"
    args.search_space = "mlp_sp"
    args.num_labels = 2
    args.device = "cpu"
    args.batch_size = 8
    args.dataset = "frappe"
    args.base_dir = "/project/exp_data/"
    args.result_dir = "/project/TRAILS/internal/ml/model_selection/exp_result/"

    os.environ.setdefault("log_logger_folder_name", "/project/TRAILS/log_score_time_frappe")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.logger import logger
    from src.eva_engine.run_ms import RunModelSelection

    logger.info(encode_str)
    params = json.loads(encode_str)
    budget = float(params["budget"])
    score_time_per_model = float(params["score_time_per_model"])
    train_time_per_epoch = float(params["train_time_per_epoch"])
    only_phase1 = True if params["only_phase1"].lower() == "true" else False

    logger.info(budget, score_time_per_model, train_time_per_epoch, only_phase1)
    rms = RunModelSelection(args.search_space, args, is_simulate=True)
    K, U, N = rms.coordination(
        budget=budget,
        score_time_per_model=score_time_per_model,
        train_time_per_epoch=train_time_per_epoch,
        only_phase1=only_phase1)

    return orjson.dumps(
        {
            "k": K,
            "u": U,
            "n": N,
        }).decode('utf-8')


@exception_catcher
def filtering_phase(encoded_str: str):
    # define dataLoader, and sample a mini-batch

    args = parse_arguments()
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    args.log_name = "score_based_all_metrics"
    args.search_space = "mlp_sp"
    args.num_labels = 2
    args.device = "cpu"
    args.batch_size = 8
    args.dataset = "frappe"
    args.base_dir = "/project/exp_data/"
    args.result_dir = "/project/TRAILS/internal/ml/model_selection/exp_result/"

    os.environ.setdefault("log_logger_folder_name", "/project/TRAILS/log_score_time_frappe")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.logger import logger
    from src.eva_engine.run_ms import RunModelSelection

    logger.info(encoded_str)
    params = json.loads(encoded_str)
    mini_batch_m = params["mini_batch_m"]
    n = int(params["n"])
    k = int(params["k"])

    logger.info(f"begin run filtering phase at {os.getcwd()}")
    mini_batch_data = json.loads(mini_batch_m)

    dataloader = LibsvmDataset(mini_batch_data)

    rms = RunModelSelection(args.search_space, args, is_simulate=True)
    k_models = rms.filtering_phase(N=n, K=k, train_loader=dataloader)

    # mini_batch_data = mini_batch.decode('utf-8')
    return orjson.dumps(
        {"k_models": k_models}).decode('utf-8')


@exception_catcher
def refinement_phase(mini_batch_m: str):
    pass
