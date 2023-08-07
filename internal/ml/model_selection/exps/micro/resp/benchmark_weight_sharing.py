import calendar
import os
import time
import torch
from exps.shared_args import parse_arguments

args = parse_arguments()

# set the log name
gmt = time.gmtime()
ts = calendar.timegm(gmt)

os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
os.environ.setdefault("log_file_name", f"{args.log_name}_{args.dataset}_wkid_{args.worker_id}_{ts}.log")
os.environ.setdefault("base_dir", args.base_dir)

from src.logger import logger
from src.eva_engine.phase2.algo.trainer import ModelTrainer
from src.search_space.init_search_space import init_search_space
from src.dataset_utils.structure_data_loader import libsvm_dataloader

# global search space
search_space_ins = init_search_space(args)
search_space_ins.load()


def load_data():
    logger.info(f" Loading data....")
    train_loader, val_loader, test_loader = libsvm_dataloader(
        args=args,
        data_dir=os.path.join(args.base_dir, "data", "structure_data", args.dataset),
        nfield=args.nfield,
        batch_size=args.batch_size)
    return train_loader, val_loader, test_loader


def train_super_net(model_path: str):
    # 1. data loader
    train_loader, val_loader, test_loader = load_data()
    # 2. train model
    model = search_space_ins.new_architecture("512-512-512-512").to(args.device)
    model.init_embedding(requires_grad=True)
    ModelTrainer.fully_train_arch(
        model=model,
        use_test_acc=False,
        epoch_num=args.epoch,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        args=args)
    # 3. save to disk
    print("training 512-512-512-512 done, save to disk")
    torch.save(model.state_dict(), model_path)


def load_model_weight_share_nas(model_path: str):
    # 1. load data
    train_loader, val_loader, test_loader = load_data()

    # 2. load model
    model = search_space_ins.new_architecture("512-512-512-512").to(args.device)
    model.init_embedding(requires_grad=False)
    model.load_state_dict(torch.load(model_path))

    # 3. evaluate
    valid_auc, _, _ = ModelTrainer.fully_evaluate_arch(
        model=model,
        use_test_acc=False,
        epoch_num=args.epoch,
        val_loader=val_loader,
        test_loader=test_loader,
        args=args)


if __name__ == "__main__":
    _model_path = f'{args.result_dir}/model_512_512_512_512.pth'
    if os.path.exists(_model_path):
        load_model_weight_share_nas(_model_path)
    else:
        train_super_net(_model_path)
