import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class QModel(nn.Module):
  def __init__(self, state_dim, action_dim, hidden):
    super().__init__()

    self.net = nn.Sequential(
      nn.Linear(state_dim, hidden),
      nn.Tanh(),
      nn.Linear(hidden, hidden),
      nn.Tanh(),
      nn.Linear(hidden, action_dim),
      nn.ReLU()
    )

  def forward(self, x):
    return self.net(x)