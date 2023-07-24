


import orjson

def filtering_phase(a: str):
    import orjson
    return orjson.dumps(
        {"data": "adsf" + a,
         "types": "adsf"}).decode('utf-8')


#
# # this is the main function of model selection.
#
# import calendar
# import os
# import time
# from src.common.constant import Config
# from src.dataset_utils.structure_data_loader import libsvm_dataloader
# from exps.shared_args import parse_arguments
# args = parse_arguments()
#
# # set the log name
# gmt = time.gmtime()
# ts = calendar.timegm(gmt)
# os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
# os.environ.setdefault("base_dir", args.base_dir)
#
# from src.eva_engine.run_ms import RunModelSelection
# from src.dataset_utils import dataset
#
#
#
#
# def filtering_phase(a: str):
#     import orjson
#     return orjson.dumps(
#         {"data": "adsf",
#          "types": "adsf"}).decode('utf-8')
#
#
# def refinement_phase():
#     pass
#
#
# def coordinator():
#     pass
#
#
# def generate_data_loader():
#     if args.dataset in [Config.c10, Config.c100, Config.imgNet]:
#         train_loader, val_loader, class_num = dataset.get_dataloader(
#             train_batch_size=args.batch_size,
#             test_batch_size=args.batch_size,
#             dataset=args.dataset,
#             num_workers=1,
#             datadir=os.path.join(args.base_dir, "data"))
#         test_loader = val_loader
#     else:
#         train_loader, val_loader, test_loader = libsvm_dataloader(
#             args=args,
#             data_dir=os.path.join(args.base_dir, "data", "structure_data", args.dataset),
#             nfield=args.nfield,
#             batch_size=args.batch_size)
#         class_num = args.num_labels
#
#     return train_loader, val_loader, test_loader, class_num
#
#
