import torch
import torch.nn as nn


class GateNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(GateNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
    
    def forward(self, x):
        """Args:
        x: [batch_size, input_size]
        Returns:
        gates: [batch_size, expert_size]
        """
        pass


class VerticalGate(GateNetwork):
    
    def __init__(self, input_size, output_size, hidden_size):
        super(VerticalGate, self).__init__(input_size, output_size, hidden_size)
        
        self.w_gate = nn.Parameter(torch.zeros(self.input_size, self.output_size), requires_grad=True)
        nn.init.xavier_normal_(self.w_gate)
    
    def forward(self, x):
        """
        Args:
        x: [batch_size, input_size]
        Returns:
        gate_score: [batch_size, output_size]
        """
        return x @ self.w_gate
    

class SparseVerticalGate(GateNetwork):
    
    def __init__(self, input_size:int, output_size:int, hidden_size:int, dropout = 0):
        super(SparseVerticalGate, self).__init__(input_size, output_size, hidden_size)
        
        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.r1 = nn.ReLU()
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        """
        Args:
        x: [batch_size, input_size]
        Returns:
        gate_score: [batch_size, output_size]
        """
        x = self.d1(self.r1(self.bn1(self.l1(x))))
        return self.l2(x)