import torch
import torch.nn as nn

class ALiBiPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(ALiBiPositionEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.alibi = self._generate_alibi(max_len, d_model)

    def _generate_alibi(self, max_len, d_model):
        alibi = torch.arange(max_len).unsqueeze(0).repeat(d_model, 1).float()
        return alibi

    def forward(self, x):
        seq_len = x.size(1)
        alibi = self.alibi[:, :seq_len].unsqueeze(0)
        return x + alibi

# Test code
if __name__ == "__main__":
    d_model = 512
    max_len = 100
    pe = ALiBiPositionEncoding(d_model, max_len)
    x = torch.zeros(1, max_len, d_model)
    output = pe(x)
    print(output.shape)  # Expected output: torch.Size([1, 100, 512])
