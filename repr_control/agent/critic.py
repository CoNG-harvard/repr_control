"""
We adapt the code from https://github.com/denisyarats/pytorch_sac
"""


import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from repr_control.utils import util


class DoubleQCritic(nn.Module):
  """Critic network, employes double Q-learning."""
  def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
    super().__init__()

    self.Q1 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
    self.Q2 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

    self.outputs = dict()
    self.apply(util.weight_init)

  def forward(self, obs, action):
    assert obs.size(0) == action.size(0)

    obs_action = torch.cat([obs, action], dim=-1)
    q1 = self.Q1(obs_action)
    q2 = self.Q2(obs_action)

    self.outputs['q1'] = q1
    self.outputs['q2'] = q2

    return q1, q2



class DoubleVCritic(nn.Module):
  """Critic network, employes double V-learning."""
  def __init__(self, obs_dim, hidden_dim, hidden_depth, use_ortho_init = True):
    super().__init__()

    self.Q1 = util.mlp(obs_dim, hidden_dim, 1, hidden_depth)
    self.Q2 = util.mlp(obs_dim, hidden_dim, 1, hidden_depth)

    # self.Q1 = util.mlp_relu(obs_dim, hidden_dim, 1, hidden_depth)
    # self.Q2 = util.mlp_relu(obs_dim, hidden_dim, 1, hidden_depth)

    self.outputs = dict()
    if use_ortho_init == True:
      self.apply(util.weight_init)

  def forward(self, obs):
    # assert obs.size(0) == action.size(0)

    q1 = self.Q1(obs)
    q2 = self.Q2(obs)

    self.outputs['q1'] = q1
    self.outputs['q2'] = q2

    return q1, q2