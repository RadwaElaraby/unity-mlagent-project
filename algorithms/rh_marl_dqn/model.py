import numpy as np
import torch

class QNetwork(torch.nn.Module):
  def __init__(self, input_size: int, encoding_size: int, output_size: int):
    super(QNetwork, self).__init__()
    self.dense1 = torch.nn.Linear(input_size, encoding_size)
    self.dense2 = torch.nn.Linear(encoding_size, output_size)
    self.softmax = torch.nn.Softmax(1)

    self.dense3 = torch.nn.Linear(input_size, output_size)

  def forward(self, obs: torch.tensor):
    hidden = self.dense1(obs)
    hidden = torch.relu(hidden)
    hidden = self.dense2(hidden)
    hidden = self.softmax(hidden)
    #hidden = self.dense3(obs)
    #hidden = self.softmax(hidden)
    return hidden