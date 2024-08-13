import numpy as np
import torch
import torch.nn as nn

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Test code
if __name__ == "__main__":
    d_model = 512
    max_len = 100
    pe = SinusoidalPositionEncoding(d_model, max_len)
    x = torch.zeros(max_len, 1, d_model)
    output = pe(x)
    print(output.shape)  # Expected output: torch.Size([100, 1, 512])
