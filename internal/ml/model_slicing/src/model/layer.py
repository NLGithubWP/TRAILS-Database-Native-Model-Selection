
import torch.nn as nn
from typing import Dict, Optional, Tuple
import torch


# Embedding layer
class Embedding(nn.Module):

    def __init__(self, nfeat, nemb):
        """
        :param nfeat: number of features,
            for sql, it == sum of unique value + 1 of each column
            for dataset, it == pre-defined features.
        :param nemb: hyperparameter.
        """
        super().__init__()
        self.embedding = nn.Embedding(nfeat, nemb)
        nn.init.xavier_uniform_(self.embedding.weight)


    def forward(self, x:torch.Tensor, v:Optional[torch.Tensor] = None):
        """
        :param x:
                {'id': LongTensor B*nfield,
                 'value': FloatTensor B*nfield}
        :return:    embeddings B*nfield*nemb
        """
        # convert feature to embedding.
        emb = self.embedding(x)                           # B*nfield*nemb
        # scale by value
        if v is not None:
            emb = emb * v.unsqueeze(2)                  # B*nfield*nemb
        return emb


# MLP layer
class MLP(nn.Module):

    def __init__(self, ninput, nlayers, nhid, dropout, noutput):
        """
        :param ninput: input dimension
        :param nlayers: # hidden layer
        :param nhid: # hidden layer output
        :param dropout: dropout rate
        :param noutput: output dimension
        """
        super().__init__()
        layers = list()
        for i in range(nlayers):
            layers.append(nn.Linear(ninput, nhid))
            layers.append(nn.BatchNorm1d(nhid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            ninput = nhid
        if nlayers == 0:
            nhid = ninput
        layers.append(nn.Linear(nhid, noutput))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:   FloatTensor B*ninput
        :return:    FloatTensor B*nouput
        """
        return self.mlp(x)


# MOELayer
class MOELayer(nn.Module):

    # shared across MOElayer instances.
    duplayers = 0

    def __init__(self, ninput, nhid, dropout):
        """
        :param ninput: input dimension
        :param nhid: # hidden layer output
        :param dropout: dropout rate
        """
        if MOELayer.duplayers == 0:
            raise "please set hyper K (number of duplication) "

        # dynamically update this.
        self.arch_weight = None     # B*1*K where K is the replication factor.

        super().__init__()
        self.layer_group = []
        layer_group = []
        for i in range(MOELayer.duplayers):
            layers = list()
            layers.append(nn.Linear(ninput, nhid))
            layers.append(nn.BatchNorm1d(nhid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            layer_group.append(nn.Sequential(*layers))
        self.layer_group = nn.Sequential(*layer_group)

    def forward(self, x):
        """
        Reconstruct layer weight and Forward
        :param x: embedding output, B*ninput
        :return: FloatTensor B*nouput
        """
        # todo: opt, skip the i-th layer computation when weight_i = 0
        # 1. generate the output for each layer
        output = list()
        for layer in self.layer_group:
            each_layer_out = layer(x)         # B * nhid
            output.append(each_layer_out)     # K * (B * nhid)
        # stack the list along a new dimension to create a tensor of shape B*K*nhid
        group_layer_output = torch.stack(output, dim=1)

        if self.arch_weight is None:
            return torch.mean(group_layer_output, dim = 1)
        # 2. weighted sum
        weighted_output = torch.bmm(self.arch_weight, group_layer_output).squeeze(1)  # B * nhid
        return weighted_output


# MOEMLP
class MOEMLP(nn.Module):

    def __init__(self, ninput, nlayers, nhid, K, dropout, noutput=1):
        """
        :param ninput: input dimension
        :param nlayers: # hidden MOELayer
        :param nhid: output dimension of hidden MOELayer
        :param dropout: dropout rate
        :param K: # duplicat number
        :param noutput: output dimension of the MOENet
        """
        # set the class variable.
        MOELayer.duplayers = K
        super().__init__()
        # build MOELayer
        self.layers = list()
        for i in range(nlayers):
            self.layers.append(MOELayer(ninput, nhid, dropout))
            ninput = nhid

        # last layer is not MOELayer
        self.layers.append(nn.Linear(nhid, noutput))
        self.moe_net = nn.Sequential(*self.layers)

    def forward(self, x, arch_weights = None):
        """
        :param x:   FloatTensor B*ninput
        :return:    FloatTensor B*nouput
        """

        if arch_weights is None:
            return self.moe_net(x)
            
        # arch weights don't affect last linear layer
        assert arch_weights.shape[1] == len(self.layers) - 1

        # update each moe layer's weight
        for index in range(arch_weights.shape[1]):
            if isinstance(self.layers[index], MOELayer):
                # sub_tensor has shape (B, 1, K)
                self.layers[index].arch_weight = arch_weights[:, index, :].unsqueeze(1)

        # then compute
        y = self.moe_net(x)  # B * 1
        return y
    


class HyperNet(torch.nn.Module):
    """
    Model:  Deep Neural Networks
    """
    def __init__(self,
                 nfield: int, nfeat: int,
                 nemb: int, nlayers: int, hid_layer_len: int,
                 dropout: float,
                 L: int, K: int):
        """
        :param nfield: # columns
        :param nfeat: # filter combinations. 'select...where...'
                      opt: sum of cardinality + 1 of each column
        :param nemb: hyperm, embedding size 10
        :param nlayers: hyperm 3
        :param hid_layer_len: hyperm, hidden layer length
        :param dropout: hyperm 0
        :param L: # layers in MOENet
        :param K: # replication for each MOELayer
        """
        super().__init__()
        # L*K used for MOENET
        noutput = L*K
        self.embedding = Embedding(nfeat, nemb)
        self.mlp_ninput = nfield * nemb
        self.mlp = MLP(self.mlp_ninput, nlayers, hid_layer_len, dropout, noutput)

    def forward(self, x: torch.Tensor):
        """
        :param x:   LongTensor B*nfield SQL Embedding
        :return:    B*L*K
        """
        x_emb = self.embedding(x)          # B*nfield*nemb
        y = self.mlp(x_emb.view(-1, self.mlp_ninput))  # B*L*K
        return y


class MOENet(torch.nn.Module):
    """
    Model:  Deep Neural Networks
    """
    def __init__(self,
                 nfield: int, nfeat: int,
                 nemb: int, moe_num_layers: int, moe_hid_layer_len: int,
                 dropout: float,
                 K: int
                 ):
        """
        :param nfield: # columns
        :param nfeat: # feature of the dataset.
        :param nemb: hyperm, embedding size 10
        :param moe_num_layers: hyperm: # hidden MOElayers of MOENet
        :param moe_hid_layer_len: hyperm: hidden layer length in MoeLayer
        :param dropout: hyperparams 0
        :param K: # duplicat number
        """
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.moe_mlp_ninput = nfield * nemb

        self.moe_mlp = MOEMLP(ninput=self.moe_mlp_ninput,
                              nlayers=moe_num_layers,
                              nhid=moe_hid_layer_len,
                              K=K, dropout=dropout)

    def forward(self, x:Tuple[torch.Tensor], arch_weights: Optional[torch.Tensor]):
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
            arch_weights=arch_weights,                # B*1*K
        )   # B*1
        
        return y.squeeze(1)

