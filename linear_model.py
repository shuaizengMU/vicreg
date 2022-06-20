import torch
import torch.nn as nn

class LinearNetwork(nn.Module):

  def __init__(self, input_dim=441, output_dim=10):
    super(LinearNetwork, self).__init__()
    self.flatten = nn.Flatten()
    self.linear = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear(x)
    return logits
