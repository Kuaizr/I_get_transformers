import torch
import torch.nn as nn

class AdaptiveSoftmax(nn.Module):
    def __init__(self, input_dim, cutoff, div_value=4.0):
        super(AdaptiveSoftmax, self).__init__()
        self.cutoff = cutoff
        self.div_value = div_value
        self.head = nn.Linear(input_dim, cutoff[0])
        self.tail = nn.ModuleList()
        for i in range(1, len(cutoff)):
            seq = nn.Sequential(
                nn.Linear(input_dim, input_dim // div_value ** i),
                nn.Linear(input_dim // div_value ** i, cutoff[i] - cutoff[i-1])
            )
            self.tail.append(seq)

    def forward(self, input):
        head_output = self.head(input)
        tail_output = [tail(input) for tail in self.tail]
        return head_output, tail_output
