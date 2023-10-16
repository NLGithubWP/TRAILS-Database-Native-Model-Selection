import os
import shutil
from torch.utils.tensorboard import SummaryWriter



def setup_tensorboard_writer(args):
    train_dir = os.path.join(args.exp,  args.dataset, args.train_dir)
    
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
        
    file_path = os.path.join(train_dir, 'args.txt')
    with open(file_path, 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    
    tensor_dir_path = os.path.join(train_dir, 'tensor_log')
    
    if os.path.exists(tensor_dir_path):
        print("Delete the exist tensorboard directiory")
        shutil.rmtree(tensor_dir_path, ignore_errors=True)

    # suffix = f"{} "
    writer = SummaryWriter(log_dir=tensor_dir_path)
    return writer   