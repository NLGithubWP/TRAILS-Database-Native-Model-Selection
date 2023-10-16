import os
import random as ra
import torch
import argparse
import numpy as np
from thop import profile, clever_format
from torch.utils.data import DataLoader
from src.model.sparsemax_verticalMoe import SliceModel, SparseMax_VerticalSAMS

import third_party.utils.func_utils as utils
from src.data_loader import sql_attached_dataloader

from src.model.factory import initialize_model

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import parameter_count_table

def load_model(tensorboard_path: str, device:str ="cuda"):
    """
    Args:
    tensorboard_path: the path of the directory of tensorboard
    """
    arg_file_path = os.path.join(tensorboard_path, "args.txt")
    model_config = reload_argparse(arg_file_path)
    
    net = initialize_model(model_config)
    
    model_pth_path = os.path.join(tensorboard_path, "best_model.pth")
    saved_state_dict = torch.load(model_pth_path, map_location=device)

    net.load_state_dict(saved_state_dict)
    print("successfully load model")
    return net, model_config

def reload_argparse(file_path:str):
    
    d = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, value = line.strip('\n').split(',')
            # print(f"{key}, {value}\n")
            try:
                re = eval(value)
            except:
                re = value
            d[key] = re
    
    return argparse.Namespace(**d)   


parser = argparse.ArgumentParser(description='predict FLOPS')
parser.add_argument('path', type=str, 
                    help="directory to model file")
parser.add_argument('--flag', '-p', action='store_true',
                    help="wehther to print profile")
parser.add_argument('--print_net', '--b', action='store_true',
                    help="print the structure of network")

parser.add_argument('--device', type=str, default="cuda",
                    help="gpu or cpu")

if __name__ == '__main__':
    args = parser.parse_args()
    path = args.path
    flag = args.flag
    device = args.device
    print(path)
    net, config= load_model(path, args.device)
    net:SparseMax_VerticalSAMS = net
    config.workload = 'random'
    
    print(config.workload)
    
    if config.net == "sparsemax_vertical_sams":
        alpha = net.sparsemax.alpha
        print(alpha)
        # net.sparsemax.alpha = torch.nn.Parameter(torch.tensor(1.6, dtype=torch.float32), requires_grad=True)
        
    train_loader, _, workload = sql_attached_dataloader(config)
    # print(net)
    net.eval()
    auc_avg_, pr_avg_ = utils.AvgrageMeter(), utils.AvgrageMeter()
    net = net.to(device)
    cnt = 0
    params = 0
    with torch.no_grad():
        workload_ops = 0
        for i in range(len(workload)):
            
            dataset, sql = workload[i]
            data_laoder = DataLoader(dataset, batch_size = 1024, shuffle = False,pin_memory=True )
            tuple_size = len(dataset)
            
            
            sql = sql.unsqueeze(0) # [1, nfield]
            sql = sql.to(device)
            
            if config.net == "sparsemax_vertical_sams":
                subnet:SliceModel = net.tailor_by_sql(sql)
                subnet.to(device)
            else:
                subnet = net
                
            subnet.eval()
            
            target_list, y_list = [], []
            ops = 0

            for _, data_batch in enumerate(data_laoder):
                target = data_batch['y'].to(device)
                target_list.append(target)
                x_id = data_batch['id'].to(device)
                B = target.shape[0]
                
                # fvcore
                if flag:
                    flops = FlopCountAnalysis(subnet, (x_id, _))
                    print(flops.by_module())
                    print(flops.total())
                    # exit(0)
                # FLops
                macs, params = profile(subnet, inputs=(x_id, _, ), verbose=flag)
                ops += macs
                # print(params)
                y = subnet(x_id, _)
                
                if isinstance(y, tuple):
                    y, _ = y
                    
                y_list.append(y)
            
            
            target = torch.cat(target_list, dim = 0) # [#n]
            y = torch.cat(y_list, dim = 0)
            
            workload_ops += ops
            ops = clever_format([ops], "%.3f")
            try:
                cnt += 1          
                auc = utils.roc_auc_compute_fn(y, target)
                auc_avg_.update(auc, 1)
                print(f"Workload {i}, #tuple {tuple_size}: AUCROC {auc:8.4f}, FLOPs {ops}")
            except ValueError:
                print("Skip this workload micro-evaluation")
                pass
        
        if args.print_net:
            print(parameter_count_table(subnet, 6))
            
        workload_ops /= len(workload)
        workload_ops = clever_format([workload_ops], "%.2f")
        
        print(f'\n--------------------Workload Test, without SQL Predicate ----------------------\n'
            f"Workload Average Ops {workload_ops} Params {params}\n"
            f'With SQL    -> Micro-AUC-ROC {auc_avg_.avg:8.4f}\n'
            )