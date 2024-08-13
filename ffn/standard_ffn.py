import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(StandardFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Test code
if __name__ == "__main__":
    ffn = StandardFFN(d_model=512, d_ff=2048)
    x = torch.randn(64, 10, 512)  # batch_size=64, seq_len=10, d_model=512
    output = ffn(x)
    print(output.shape)  # Should be (64, 10, 512)
