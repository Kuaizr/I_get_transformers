import torch
import torch.nn as nn

class GLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(GLU, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x = x1 * torch.sigmoid(x2)  # Gated Linear Unit
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# Test code
if __name__ == "__main__":
    glu = GLU(d_model=512, d_ff=2048)
    x = torch.randn(64, 10, 512)  # batch_size=64, seq_len=10, d_model=512
    output = glu(x)
    print(output.shape)  # Should be (64, 10, 512)
