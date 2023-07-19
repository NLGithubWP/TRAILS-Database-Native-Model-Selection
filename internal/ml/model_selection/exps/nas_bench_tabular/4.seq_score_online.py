import calendar
import json
import os
import random
import time
from exps.shared_args import parse_arguments
from datetime import datetime


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


def run_with_time_budget(time_budget: float):
    """
    :param time_budget: the given time budget, in second
    :return:
    """

    # define dataLoader, and sample a mini-batch
    train_loader, val_loader, test_loader, class_num = generate_data_loader()
    args.num_labels = class_num
    data_loader = [train_loader, val_loader, test_loader]

    rms = RunModelSelection(args.search_space, args, is_simulate=True)
    best_arch, best_arch_performance, _, _, _ = rms.select_model_online(time_budget, data_loader)

    return best_arch, best_arch_performance


if __name__ == "__main__":
    args = parse_arguments()

    random.seed(80)
    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.common.constant import Config
    from src.common.structure import ModelAcquireData
    from src.controller.sampler_all.seq_sampler import SequenceSampler
    from src.eva_engine.phase1.evaluator import P1Evaluator
    from src.logger import logger
    from src.search_space.init_search_space import init_search_space
    from src.storage.structure_data_loader import libsvm_dataloader
    from src.utilslibs.io_tools import write_json, read_json
    from src.eva_engine.run_ms import RunModelSelection
    from src.storage import dataset

    search_space_ins = init_search_space(args)
    search_space_ins.load()
    train_loader, val_loader, test_loader, class_num = generate_data_loader()

    _evaluator = P1Evaluator(device=args.device,
                             num_label=args.num_labels,
                             dataset_name=args.dataset,
                             search_space_ins=search_space_ins,
                             train_loader=train_loader,
                             is_simulate=False)

    sampler = SequenceSampler(search_space_ins)

    explored_n = 0
    output_file = f"{args.result_dir}/score_{args.dataset}_batch_size_{args.batch_size}.json"
    result = read_json(output_file)
    print(f"begin to score all, currently we already explored {len(result.keys())}")
    while True:
        arch_micro, _ = sampler.sample_next_arch()
        if arch_micro is None:
            break
        if arch_micro in result:
            continue
        # run the model selection
        model_acquire_data = ModelAcquireData(model_id=str(arch_micro),
                                              model_encoding=arch_micro,
                                              is_last=False)
        data_str = model_acquire_data.serialize_model()
        model_score = _evaluator.p1_evaluate(data_str)
        explored_n += 1
        result[arch_micro] = model_score
        print(f" {datetime.now()}finish arch = {arch_micro}")
        if explored_n % 100 == 0:
            print("3. [trails] Phase 1: filter phase explored " + str(explored_n)
                  + "Total explored" + str(len(result)) +
                  " model, model_id = " + str(arch_micro) +
                  " model_scores = " + json.dumps(model_score))
            logger.info("3. [trails] Phase 1: filter phase explored " + str(explored_n) +
                        " model, model_id = " + str(arch_micro) +
                        " model_scores = " + json.dumps(model_score))
            write_json(output_file, result)
    write_json(output_file, result)
