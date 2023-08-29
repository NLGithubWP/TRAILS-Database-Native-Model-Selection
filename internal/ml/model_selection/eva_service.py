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
from internal.ml.model_selection.shared_config import parse_config_arguments


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
