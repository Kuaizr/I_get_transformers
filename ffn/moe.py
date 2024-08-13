import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, dropout=0.1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(d_model, d_ff) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        gate_values = F.softmax(self.gate(x), dim=-1)  # Shape: (batch_size, seq_len, num_experts)
        expert_outputs = [expert(x) for expert in self.experts]  # List of (batch_size, seq_len, d_ff)
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # Shape: (batch_size, seq_len, d_ff, num_experts)
        weighted_output = torch.einsum('bse,bsef->bsf', gate_values, expert_outputs)  # Shape: (batch_size, seq_len, d_ff)
        weighted_output = F.relu(weighted_output)
        weighted_output = self.dropout(weighted_output)
        output = self.output_layer(weighted_output)
        return output

# Test code
if __name__ == "__main__":
    moe = MoE(d_model=512, d_ff=2048, num_experts=4)
    x = torch.randn(64, 10, 512)  # batch_size=64, seq_len=10, d_model=512
    output = moe(x)
    print(output.shape)  # Should be (64, 10, 512)
