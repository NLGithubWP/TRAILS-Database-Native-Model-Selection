import argparse
import torch
import torch.nn as nn
from src.model.expert import VerticalDNN, initialize_expert
from src.model.gate_network import SparseVerticalGate, VerticalGate
from src.model.sparseMoE import SparseMoE



    



class VerticalSAMS(nn.Module):
    
    def __init__(self, args:argparse.Namespace):
        super().__init__()
        self.args = args
        
        self.gate_input_size = args.nfield * args.data_nemb
        self.gate_network = SparseVerticalGate(self.gate_input_size, args.K, args.hid_layer_len, args.dropout)
        self.input_size = args.nfield * args.data_nemb
        self.hidden_size = args.hid_layer_len
        self.output_size = args.output_size
        self.num_experts = args.K
        self.select_experts = args.C
        
        assert args.C <= args.K
        expert = initialize_expert(args)
        
        self.backbone = SparseMoE( 
                                expert = expert, 
                                num_experts = self.num_experts,
                                select_experts = self.select_experts,
                                noisy_gating = args.noise_gating,
                                )
        
        self.input_embedding = nn.Embedding(args.nfeat, args.data_nemb)

        nn.init.xavier_uniform_(self.input_embedding.weight)

    
    def cal_gate_score(self, sql:torch.Tensor):
        B = sql.shape[0]
        sql_emb = self.sql_embedding(sql).view(B, -1)
        gate_score = self.gate_network(sql_emb)
        
        top_indices = torch.topk(gate_score, k=self.select_experts, dim = 1).indices
        result = torch.zeros_like(gate_score)
        result.scatter_(1, top_indices, gate_score.gather(1, top_indices))
        return result
    
    def forward(self, x:torch.Tensor, sql:torch.Tensor):
        """
        Args:
        x:      [B, nfield] only Id
        sql:    [B, nfield]
        Returns:
        y:      [B]
        loss:   [1]
        """
        if self.args.expert == "afn":
            self.embedding_clip()
            
        x_emb = self.input_embedding(x)             # [B, nfield, data_nemb]
        B = x.shape[0]
        
        x_emb_ = x_emb.view(B, -1)                   # [B, input_size]

        gate_scores = self.gate_network(x_emb_)    # [B, num_experts]
        y, loss = self.backbone(x_emb, gate_scores) 
        return y.squeeze(-1), loss
    
    def embedding_clip(self):
        "Keep AFN expert embeddings positive"
        with torch.no_grad():
            self.input_embedding.weight.abs_()
            self.input_embedding.weight.clamp_(min=1e-4)