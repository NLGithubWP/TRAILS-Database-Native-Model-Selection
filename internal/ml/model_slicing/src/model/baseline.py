import argparse
from copy import deepcopy
import torch
import torch.nn as nn

from src.model.expert import initialize_expert


class MeanMoE(nn.Module):
    
    def __init__(self, args:argparse.Namespace):
        super().__init__()
        
        self.args = args
        
        self.num_experts = args.K
        
        expert = initialize_expert(args)
        self.experts = nn.ModuleList(
            [deepcopy(expert) for _ in range(self.num_experts)]
        )
        
        self.input_embedding = nn.Embedding(args.nfeat, args.data_nemb)
        nn.init.xavier_uniform_(self.input_embedding.weight)
        
    
    def forward(self, x:torch.Tensor, sql:torch.Tensor):
        
        if self.args.expert == "afn":
            self.embedding_clip()
            
        x_emb = self.input_embedding(x)
        x_list = [expert(x_emb) for expert in self.experts]
        x = torch.stack(x_list, dim = 1)      # [B, N, 1]
        
        x = x.squeeze(-1)   #[B, N]
        x = torch.mean(x, dim = -1) #[B]

        return x
    
    def embedding_clip(self):
        "Keep AFN expert embeddings positive"
        with torch.no_grad():
            self.input_embedding.weight.abs_()
            self.input_embedding.weight.clamp_(min=1e-4)