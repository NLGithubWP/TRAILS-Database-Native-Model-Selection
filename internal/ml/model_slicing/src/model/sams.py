import torch
import torch.nn as nn
import argparse

from src.model.layer import HyperNet,MOENet
from third_party.utils.model_utils import EntmaxBisect


class SAMS(nn.Module):
    
    def __init__(self, args:argparse.ArgumentParser):
        super().__init__()

        self.args = args
        col_cardinality_sum = args.nfield + args.nfeat + 1
        self.hyper_net = HyperNet(
            nfield=self.args.nfield,
            nfeat= col_cardinality_sum,
            nemb=self.args.sql_nemb,
            nlayers=self.args.hyper_num_layers,
            hid_layer_len=self.args.hid_layer_len,
            dropout=self.args.dropout,
            L=self.args.moe_num_layers,
            K=self.args.K,
        )

        self.moe_net = MOENet(
            nfield=self.args.nfield,
            nfeat=self.args.nfeat,
            nemb=self.args.data_nemb,
            moe_num_layers=self.args.moe_num_layers,
            moe_hid_layer_len=self.args.moe_hid_layer_len,
            dropout=self.args.dropout,
            K=self.args.K,
        )
          
        if self.args.alpha == 1.:
            self.sparsemax = nn.Softmax(dim=-1)
        else:
            self.sparsemax = EntmaxBisect(self.args.alpha, dim=-1)
        
    
    
    def forward(self, x, sql):
        arch_advisor = self.hyper_net(sql)
        # (B, L * K)
        
        arch_advisor = arch_advisor.reshape(arch_advisor.size(0), -1 ,self.args.K)
        # (B, L, K)
        
        arch_advisor = self.sparsemax(arch_advisor)
        
        # x_id, x_value = x
        
        # y = self.moe_net((x_id, x_value), arch_advisor)
        
        y = self.moe_net(x, arch_advisor)
        # (B, C)
        return y
        