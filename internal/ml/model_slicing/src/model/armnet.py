from typing import Tuple
import torch
import torch.nn as nn
from src.model.expert import ARMExpert

from src.model.layer import Embedding

class ARMNet(nn.Module):
    """
    Model:  Adaptive relation modeling network
    Ref:    Cai, Shaofeng, et al. 
        "Arm-net: Adaptive relation modeling network for structured data." 
        Proceedings of the 2021 International Conference on Management of Data. 2021.
    """
    
    def __init__(self, nfield:int, nfeat:int, nemb:int, output_size:int ,hidden_size:int, 
                 nhead:int, nhid:int, alpha:float, dropout:float):
        """
        :param nfield: # columns
        :param nfeat: # feature of the dataset.
        :param nemb: hyperm, embedding size 10
        :param hid_size: hyperm: hidden layer length in MoeLayer
        :param K: # duplicat number
        :param dropout: hyperparams 0
        """
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        
        self.net = ARMExpert(nfield, nemb, output_size, hidden_size, 
                             nhead, nhid, alpha, dropout)

    def forward(self, x:Tuple[torch.Tensor], _):
        """
        :param x: {'id': LongTensor B*nfield, 'value': FloatTensor B*nfield}
        :param arch_weights: B*L*K
        :return: y of size B, Regression and Classification (+sigmoid)
        """
        # x_id, x_value = x
        # x_emb = self.embedding(x_id, x_value)         # [B,F,E]
        
        x_emb = self.embedding(x)                       # [B,F,E]
        
        y = self.net(
            x_emb
        )   # B*1
        return y.squeeze(1)