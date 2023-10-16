from typing import Tuple
import torch

from src.model.layer import Embedding, MOEMLP
from src.model.verticalMoE import VerticalDNN

class MoEDNN(torch.nn.Module):
    """
    Model:  Deep Neural Networks
    """
    def __init__(self, nfield:int, nfeat:int, nemb:int, moe_num_layers:int, 
                 moe_hid_size:int, dropout:float, K:int):
        """
        :param nfield: # columns
        :param nfeat: # feature of the dataset.
        :param nemb: hyperm, embedding size 10
        :param moe_num_layers: hyperm: # hidden MOElayers of MOENet
        :param moe_hid_size: hyperm: hidden layer length in MoeLayer
        :param dropout: hyperparams 0
        :param K: # duplicat number
        """
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.moe_mlp_ninput = nfield * nemb

        self.moe_mlp = MOEMLP(ninput=self.moe_mlp_ninput,
                              nlayers=moe_num_layers,
                              nhid=moe_hid_size,
                              K=K, dropout=dropout)
        # self.moe_mlp = VerticalDNN(self.moe_mlp_ninput, moe_num_layers)

    def forward(self, x:Tuple[torch.Tensor], _):
        """
        :param x: {'id': LongTensor B*nfield, 'value': FloatTensor B*nfield}
        :param arch_weights: B*L*K
        :return: y of size B, Regression and Classification (+sigmoid)
        """
        # x_id, x_value = x
        # x_emb = self.embedding(x_id, x_value)         # B*nfield*nemb
        x_emb = self.embedding(x)
        y = self.moe_mlp(
            x=x_emb.view(-1, self.moe_mlp_ninput),    # B*nfield*nemb
            arch_weights= None,                # B*1*K
        )   # B*1
        
        return y.squeeze(1)


class DNN(torch.nn.Module):
    """
    Model:  Deep Neural Networks
    """
    def __init__(self, nfield:int, nfeat:int, nemb:int, output_size:int,hid_size:int, dropout:float):
        """
        :param nfield: # columns
        :param nfeat: # feature of the dataset.
        :param nemb: hyperm, embedding size 10
        :param hid_size: hyperm: hidden layer length in MoeLayer
        :param dropout: hyperparams 0
        """
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)

        self.mlp = VerticalDNN(nfield, nemb, output_size, hid_size, dropout)

    def forward(self, x:Tuple[torch.Tensor], _):
        """
        :param x: {'id': LongTensor B*nfield, 'value': FloatTensor B*nfield}
        :param arch_weights: B*L*K
        :return: y of size B, Regression and Classification (+sigmoid)
        """
        # x_id, x_value = x
        # x_emb = self.embedding(x_id, x_value)         # B*nfield*nemb
        x_emb = self.embedding(x)
        y = self.mlp(
            x_emb
        )   # B*1
        
        return y.squeeze(1)