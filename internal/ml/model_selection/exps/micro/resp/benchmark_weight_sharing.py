import calendar
import os
import time
import torch
from exps.shared_args import parse_arguments

if __name__ == "__main__":
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

    search_space_ins = init_search_space(args)
    search_space_ins.load()

    # 1. data loader
    logger.info(f" Loading data....")
    train_loader, val_loader, test_loader = libsvm_dataloader(
        args=args,
        data_dir=os.path.join(args.base_dir, "data", "structure_data", args.dataset),
        nfield=args.nfield,
        batch_size=args.batch_size)

    model = search_space_ins.new_architecture("512-512-512-512").to(args.device)
    model.init_embedding(requires_grad=True)
    valid_auc, total_run_time, train_log = ModelTrainer.fully_train_arch(
        model=model,
        use_test_acc=False,
        epoch_num=args.epoch,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        args=args)

    print("training 512-512-512-512 done, save to disk")
    torch.save(model.state_dict(), f'{args.result_dir}/model_512_512_512_512.pth')
