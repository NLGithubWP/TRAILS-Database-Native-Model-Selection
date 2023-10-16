from typing import Tuple
import torch
import torch.nn as nn
from src.model.expert import AFNExpert

from src.model.layer import Embedding

class AFN(nn.Module):
    """
    Model:  Adaptive Factorization Network
    Ref:    W Cheng, et al. Adaptive Factorization Network:
                Learning Adaptive-Order Feature Interactions, 2020.
    """
    
    def __init__(self, nfield:int, nfeat:int, nemb:int, output_size:int ,hid_size:int, afn_hid_size, dropout):
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
        self.emb_bn = nn.BatchNorm1d(nfield)
        
        self.net = AFNExpert(nfield, nemb, output_size, hid_size, afn_hid_size, dropout) 

    def forward(self, x:Tuple[torch.Tensor], _):
        """
        :param x: {'id': LongTensor B*nfield, 'value': FloatTensor B*nfield}
        :param arch_weights: B*L*K
        :return: y of size B, Regression and Classification (+sigmoid)
        """
        # x_id, x_value = x
        # x_emb = self.embedding(x_id, x_value)         # [B,F,E]
        
        # keep postive for input embedding
        self.embedding_clip()
        
        x_emb = self.embedding(x)                       # [B,F,E]
        
        # transfer it into log space
        x_log = self.emb_bn(torch.log(x_emb))           # [B,F,E]
        
        y = self.net(
            x_log
        )   # B*1
        return y.squeeze(1)

    
    def embedding_clip(self):
        "keep AFN embeddings positive"
        with torch.no_grad():
            self.embedding.embedding.weight.abs_()
            self.embedding.embedding.weight.clamp_(min=1e-4)