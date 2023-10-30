import torch
import torch.nn as nn

class FashionMNISTModel(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super(FashionMNISTModel, self).__init__()
    self.layer_stack = nn.Sequential(
      nn.Flatten(),
      nn.Linear(input_shape, hidden_units),
      nn.ReLU(),
      nn.Linear(hidden_units, output_shape),
      nn.ReLU()
      )
  def forward(self, x:torch.Tensor):
    return self.layer_stack(x)