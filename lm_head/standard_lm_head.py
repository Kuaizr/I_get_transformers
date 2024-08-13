import torch.nn as nn

class StandardLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(StandardLMHead, self).__init__()
        self.dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        return self.dense(hidden_states)
