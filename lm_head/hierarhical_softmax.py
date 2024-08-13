import torch
import torch.nn as nn

class HierarchicalSoftmax(nn.Module):
    def __init__(self, input_dim, tree_structure):
        super(HierarchicalSoftmax, self).__init__()
        self.tree_structure = tree_structure
        self.layers = nn.ModuleList()
        for branch in tree_structure:
            self.layers.append(nn.Linear(input_dim, len(branch)))

    def forward(self, input):
        outputs = []
        for layer in self.layers:
            outputs.append(layer(input))
        return outputs
