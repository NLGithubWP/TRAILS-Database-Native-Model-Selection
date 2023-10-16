import torch
from src.model.expert import NFMExpert

from src.model.layer import Embedding


class NFM(torch.nn.Module):
    
    """
    Model: Neural Factorization Machine
    Ref: X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """
    
    def __init__(self, nfield:int, nfeat:int, nemb:int, output_size:int ,hidden_size:int, dropout:float):
        """
        :param nfield: # columns
        :param nfeat: # feature of the dataset.
        :param nemb: hyperm, embedding size 10
        :param hid_size: hyperm: hidden layer length in MoeLayer
        :param dropout: hyperparams 0
        """
        
        super().__init__()
        
        # Linear Part
        self.weight = Embedding(nfeat, output_size)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))
        
        self.embedding = Embedding(nfeat, nemb)
        self.net = NFMExpert(nfield, nemb, output_size, hidden_size, dropout)
        
        
    def forward(self, x, _):
        """
        x, [B,F]
        """
        x_emb = self.embedding(x)   # [B,F,E]
        nfm = self.net(x_emb)       # [B,1]
        
        li = torch.sum(self.weight(x), dim = 1) + self.bias # [B, 1]
        
        x = nfm + li
        return x.squeeze(1)