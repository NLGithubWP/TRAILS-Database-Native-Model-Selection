import argparse
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from third_party.utils.model_utils import EntmaxBisect

class Expert(nn.Module):
    def __init__(self, nfield, nemb , output_size, hidden_size):
        super(Expert,self).__init__()
        self.nfield = nfield
        self.nemb = nemb
        self.input_size = nfield * nemb
        self.output_size = output_size
        self.hidden_size = hidden_size


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.r1 = nn.ReLU()
        self.d1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.r2 = nn.ReLU()
        self.d2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.r3 = nn.Sigmoid()

        nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        # nn.init.xavier_uniform_(self.fc3.bias)
        
    def forward(self, x):
        """
        Should consider the B = 1, in SparseMoe it will dispatch input_batch
        Args:
        x : [batch_size, input_size]
        Return:
        x : [batch_size, output_size]
        """
        B, _ = x.size()
        if B == 1:
            x = self.d1(self.r1(self.fc1(x)))
            x = self.d2(self.r2(self.fc2(x)))
            x = self.fc3(x)
        else:
            x = self.d1(self.r1(self.bn1(self.fc1(x))))
            x = self.d2(self.r2(self.bn2(self.fc2(x))))
            x = self.fc3(x)
        return x
 

class VerticalDNN(Expert):
    
    def __init__(self, nfield, nemb , output_size, hidden_size, dropout):
        super(VerticalDNN, self).__init__(nfield, nemb, output_size, hidden_size)
        
        
        assert output_size == 1
        # default binary classification.
        input_size = nemb * nfield
        self.mlp = MLP(input_size, output_size, hidden_size, dropout)
        # self.r3 = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
        x : [batch_size, nfield , nfeat]
        Return:
        x : [batch_size, output_size]
        """
        B,F,E = x.size()
        x = x.view(B, F * E)  # x: [batch_size, nfield * nfeat]
        x = self.mlp(x)
        return x


class CompressedInteractionExpert(Expert):
    def __init__(self, nfield, nemb, output_size, hidden_size):
        super().__init__(nfield, nemb, output_size, hidden_size)
        """_summary_
        hidden_size: number of filter
        default layer is two
        """
        assert output_size == 1
        self.conv1 = nn.Conv1d(nfield * nfield, hidden_size, kernel_size=1, bias =False)
        
        self.r1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(nfield * hidden_size, hidden_size, kernel_size=1, bias=False)
        self.r2 = nn.ReLU()
        
        self.affine = nn.Linear(hidden_size * 2, output_size, bias=False)
    
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : [batch_size, nfield, nfeat] -> [B,F,E]
        Return:
        x : [batch_size, output_size]
        """
        xlist = list()
        
        B,F,E = x.size()
        x0, xk = x.unsqueeze(2), x         # x0: [B,F,1,E] | xk: [B,F,E]
        h = x0 * xk.unsqueeze(1)           # [B,F,1,E] * [B,1,F,E] -> h [B,F,F,E]
        B,F,H,E = h.size()
        xk = self.r1(self.conv1(h.view(B, F*H, E))) # B * H * E
        xlist.append(torch.sum(xk, dim = -1))
        
        h = x0 * xk.unsqueeze(1)           # [B,F,1,E] * [B,1,H,E] -> h [B,F,H,E]
        B,F,H,E = h.size()
        xk = self.r2(self.conv2(h.view(B, F*H, E))) # [B, H, E]
        xlist.append(torch.sum(xk, dim = -1))
        
        x = torch.cat(xlist, dim = 1)   # [B, H*2]
        y = self.affine(x)              # [B, 1]
        return y
    

class AFNExpert(Expert):
    
    def __init__(self, nfield, nemb, output_size, hidden_size, afn_hidden_size, dropout):
        super().__init__(nfield, nemb, output_size, hidden_size)

        assert output_size == 1
        self.dropout = nn.Dropout(p=dropout)

        self.afn = nn.Linear(nfield, afn_hidden_size)
        self.afn_bn = nn.BatchNorm1d(afn_hidden_size)
        nn.init.normal_(self.afn.weight, std=0.1)
        nn.init.constant_(self.afn.bias, 0.)        
        
        # 2-layer mlp
        self.mlp =  MLP(afn_hidden_size*nemb, output_size, hidden_size, dropout)
        # TODO: original AFN adopt order:   Linear->ReLU->BN->Dropout
        #  recent reserach recommend:       Linear->BN->ReLU->Dropout
        
    def forward(self, x):
        """
        Args:
        x -> [B,F,E], id,
        Embedding should be keep positive for afn model
        need embedding clip and embedding batch_norm
        """
        B,F,E = x.size()
        x_log = x.transpose(1,2) #[B,E,F]
        afn = torch.exp(self.afn(x_log))        #[B,E,H_AFN]
        
        if B != 1:
            afn = self.afn_bn(afn.transpose(1,2))   #[B,H_AFN, E]
        
        B,F,E = afn.size()
        afn = afn.view(B, F*E)                   #[B, H_AFN * E] -> [B, MLP_IN]
        
        afn = self.dropout(afn)
        y = self.mlp(afn)                       #[B, OUTPUT]
        return y
    

class SparseAttLayer(nn.Module):
    def __init__(self, nhead: int, nfield: int, nemb: int, d_k: int, nhid: int, alpha: float = 1.5):
        """ Multi-Head Sparse Attention Layer """
        super(SparseAttLayer, self).__init__()
        self.sparsemax = nn.Softmax(dim=-1) if alpha == 1. \
            else EntmaxBisect(alpha, dim=-1, requires_grad=False)

        self.scale = d_k ** -0.5
        self.bilinear_w = nn.Parameter(torch.zeros(nhead, nemb, d_k))                   # nhead*nemb*d_k
        self.query = nn.Parameter(torch.zeros(nhead, nhid, d_k))                        # nhead*nhid*d_k
        self.values = nn.Parameter(torch.zeros(nhead, nhid, nfield))                    # nhead*nhid*nfield
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.bilinear_w, gain=1.414)
        nn.init.xavier_uniform_(self.query, gain=1.414)
        nn.init.xavier_uniform_(self.values, gain=1.414)

    def forward(self, x):
        """
        :param x:   [bsz, nfield, nemb], FloatTensor
        :return:    Att_weights [bsz, nhid, nfield], FloatTensor
        """
        keys = x                                                                        # bsz*nfield*nemb
        # sparse gates
        att_gates = torch.einsum('bfx,kxy,koy->bkof',
                                 keys, self.bilinear_w, self.query) * self.scale        # bsz*nhead*nhid*nfield
        sparse_gates = self.sparsemax(att_gates)                                        # bsz*nhead*nhid*nfield
        return torch.einsum('bkof,kof->bkof', sparse_gates, self.values)


class ARMExpert(Expert):
    def __init__(self, nfield:int, nemb:int, output_size:int, hidden_size:int, 
                 nhead:int,  arm_hid_size:int, alpha:float, dropout:float):
        super().__init__(nfield, nemb, output_size, hidden_size)
        
        self.attn_layer = SparseAttLayer(nhead, nfield, nemb, nemb, arm_hid_size, alpha)
        self.arm_bn = nn.BatchNorm1d(nhead * arm_hid_size)
        
        self.mlp = MLP(nhead * arm_hid_size * nemb, output_size, hidden_size, dropout)
        
    
    def forward(self, x):
        """
        Args:
        x : [B,F,E]
        """
        B,_, _ = x.size()
        arm_weight = self.attn_layer(x)                                 # [bsz,nhead,nhid,nfield]
        x_arm = torch.exp(
            torch.einsum('bfe,bkof->bkoe', x, arm_weight))          # [bsz,nhead,nhid,nemb]
        
        x_arm = rearrange(x_arm, 'b k o e -> b (k o) e')                # [bsz, nhead * nhid, nemb]
        # b,_,_, e = x_arm.size()
        # x_arm = x_arm.view(b,-1, e)
        if B != 1:
            x_arm = self.arm_bn(x_arm)                                      # [bsz, nhead * nhid, nemb]
        
        x_arm = rearrange(x_arm, 'b h e -> b (h e)')                    # [bsz, nhead*nhid*nemb]
        # b,_,_ = x_arm.size()
        # x_arm = x_arm.view(b,-1)
        y = self.mlp(x_arm)                                             # [bsz, output]
        
        return y
    
    
class NFMExpert(Expert):
    def __init__(self, nfield:int, nemb:int, output_size:int, hidden_size:int, dropout:float):
        super().__init__(nfield, nemb, output_size, hidden_size)
        self.b1 = nn.BatchNorm1d(nemb)
        self.d = nn.Dropout(dropout)
    
        self.mlp = MLP(nemb, output_size, hidden_size, dropout)
        
        
    def forward(self, x):
        """
        x: [B,F,E]
        """
        
        """
        Neural Factorization Machine
        """
        square_of_sum = torch.sum(x, dim = 1)**2         # [B, E]
        sum_of_square = torch.sum(x**2, dim = 1)         # [B, E]
        fm = 0.5* (square_of_sum - sum_of_square)        # [B, E]
        fm = self.d(self.b1(fm))
        return self.mlp(fm)


class AFMExpert(Expert):
    
    def __init__(self, nfield, nemb, output_size, hidden_size,
                 nhid:int, dropout:float):
        super().__init__(nfield, nemb, output_size, hidden_size)
        self.attn_w = nn.Linear(nemb, nhid)
        self.attn_h = nn.Linear(nhid, output_size)
        self.attn_p = nn.Linear(nemb, 1)
        
        self.d  = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x:[B,F,E]
        """
        B,F,E = x.size()
        vi_indices, vj_indices = np.triu_indices(F, 1)
        vi, vj = x[:, vi_indices], x[:, vj_indices]                 # B*(Fx(F-1)/2)*E
        hadamard_prod = vi * vj
        attn_weights = nn.functional.relu(self.attn_w(hadamard_prod))           # B*(Fx(F-1)/2)*nattn
        attn_weights = nn.functional.softmax(self.attn_h(attn_weights), dim=1)  # B*(Fx(F-1)/2)*1
        attn_weights = self.d(attn_weights)
        afm = torch.sum(attn_weights*hadamard_prod, dim=1)          # B*E
        afm = self.attn_p(self.d(afm))
        return afm

        
def initialize_expert(args:argparse.Namespace):
    
    if args.expert == "dnn":
        return VerticalDNN(args.nfield, args.data_nemb, args.output_size, 
                                    args.moe_hid_layer_len, args.dropout)
    
    if args.expert == "cin":
        return CompressedInteractionExpert(args.nfield, args.data_nemb, args.output_size,
                                    args.nhid)
    
    if args.expert == "afn":
        return AFNExpert(args.nfield, args.data_nemb, args.output_size,
                                args.moe_hid_layer_len, args.nhid, args.dropout)
    
    if args.expert == "armnet":
        return ARMExpert(args.nfield, args.data_nemb, args.output_size,
                                args.moe_hid_layer_len, args.nhead, args.nhid, 2, args.dropout)
                            # alpha = 2.0 sparseMax
    
    if args.expert == "nfm":
        return NFMExpert(args.nfield, args.data_nemb, args.output_size, 
                         args.moe_hid_layer_len, args.dropout)
    
    if args.expert == "afm":
        return AFMExpert(args.nfield, args.data_nemb, args.output_size,
                         args.moe_hid_layer_len, args.nhid, args.dropout)

    