from typing import Tuple
import torch
from src.model.expert import CompressedInteractionExpert

from src.model.layer import Embedding



class CIN(torch.nn.Module):
    """
    Model:  CIN (w/o a neural network)
    Ref:    J Lian, et al. xDeepFM: Combining Explicit and
                Implicit Feature Interactions for Recommender Systems, 2018.
    """
    def __init__(self, nfield:int, nfeat:int, nemb:int, output_size:int ,hid_size:int):
        """
        :param nfield: # columns
        :param nfeat: # feature of the dataset.
        :param nemb: hyperm, embedding size 10
        :param hid_size: hyperm: hidden layer length in MoeLayer
        :param dropout: hyperparams 0
        """
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.net = CompressedInteractionExpert(nfield, nemb, output_size, hid_size)

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