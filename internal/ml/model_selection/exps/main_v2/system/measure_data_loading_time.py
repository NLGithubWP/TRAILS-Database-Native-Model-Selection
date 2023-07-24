

import time
from storage.structure_data_loader import libsvm_dataloader_ori
from exps.main_v2.common.shared_args import parse_arguments

args = parse_arguments()

begin_time = time.time()
train_loader, val_loader, test_loader = libsvm_dataloader_ori(args=args)

print(time.time() - begin_time)

