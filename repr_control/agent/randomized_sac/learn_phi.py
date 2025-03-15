import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from repr_control.utils import util

class phiNet(nn.Module):
  """phi(s,a) network"""
  def __init__(self, obs_dim, action_dim, hidden_dim, output_dim, hidden_depth):
    super().__init__()

    self.nn = util.mlp(obs_dim + action_dim, hidden_dim, output_dim, hidden_depth)
   

    self.outputs = dict()
    self.apply(util.weight_init)

  def forward(self, obs, action):
    assert obs.size(0) == action.size(0)

    # print("obs", obs)
    obs_action = torch.cat([obs, action], dim=-1)
    phi = self.nn(obs_action)

    return phi
  
class muNet(nn.Module):
  """mu(s') network"""
  def __init__(self, obs_dim, hidden_dim, output_dim, hidden_depth):
    super().__init__()

    self.nn = util.mlp(obs_dim, hidden_dim, output_dim, hidden_depth)
   

    # self.outputs = dict()
    self.apply(util.weight_init)

  def forward(self, obs):

    mu = self.nn(obs)

    return mu


  # nn Mu that takes in only s.
class nnMu(nn.Module):
	def __init__(self, state_dim, output_dim=512, hidden_dim=256, hidden_depth=2):
		super().__init__()
		self.model = util.mlp(
			input_dim=state_dim,
			hidden_dim=hidden_dim,
			output_dim=output_dim,
			hidden_depth=hidden_depth
		)
	
	def forward(self, state):
		phi = self.model(state)
		return phi

class randMu(nn.Module):
  """mu(s') random function"""
  def __init__(self, obs_dim, output_dim, sigma = 1.):
    super().__init__()
    fourier_feats = nn.Linear(obs_dim, output_dim)
    init.normal_(fourier_feats.weight, std = 1./sigma)
    init.uniform_(fourier_feats.bias, 0, 2 * np.pi)
    fourier_feats.weight.requires_grad = False
    fourier_feats.bias.requires_grad = False
    self.fourier = fourier_feats
    print("self.fourier weights", self.fourier.weight)

  def forward(self, states:torch.Tensor):
    output = torch.cos(self.fourier(states))
    return output

class randMu2(nn.Module):
  """mu(s') random function, trying to sample from posterior"""
  def __init__(self, obs_dim, rf_dim, output_dim, sigma = 1.):
    super().__init__()
    fourier_feats = nn.Linear(obs_dim, rf_dim)
    init.normal_(fourier_feats.weight, std = 1./sigma)
    init.uniform_(fourier_feats.bias, 0, 2 * np.pi)
    fourier_feats.weight.requires_grad = False
    fourier_feats.bias.requires_grad = False
    self.fourier = fourier_feats
    print("self.fourier weights", self.fourier.weight)
    rand_weights = nn.Linear(rf_dim, output_dim)
    init.normal_(rand_weights.weight, std = 1.0)
    init.constant_(rand_weights.bias, 0)
    rand_weights.weight.requires_grad = False
    rand_weights.bias.requires_grad = False
    self.rand_weights = rand_weights
    self.rf_dim = rf_dim

  def forward(self, states:torch.Tensor):
    output = math.sqrt(1./self.rf_dim) * self.rand_weights(torch.cos(self.fourier(states)))
    return output

class randPhi(nn.Module):
  """phi(s,a) random function"""
  def __init__(self, obs_dim, action_dim, output_dim, dynamics_fn = None, action_range = None,sigma = 1.,device = 'cpu'):
    super().__init__()
    # fourier_feats = nn.Linear(obs_dim + action_dim, output_dim)
    fourier_feats = nn.Linear(obs_dim, output_dim)
    init.normal_(fourier_feats.weight, std = 1./sigma)
    init.uniform_(fourier_feats.bias, 0, 2 * np.pi)
    fourier_feats.weight.requires_grad = False
    fourier_feats.bias.requires_grad = False
    self.action_low = torch.tensor(action_range[0], device = device)
    self.action_high = torch.tensor(action_range[1], device = device)
    self.fourier = fourier_feats
    self.dynamics_fn = dynamics_fn

  def rescale_action(self, actions):
      """
      rescale action from [-1, 1] to action range.
      """
      return self.action_low.unsqueeze(dim=0) + (self.action_high - self.action_low).unsqueeze(dim=0) * (actions + 1) * 0.5

  def forward(self, states:torch.Tensor, actions:torch.Tensor):
    # obs_action = torch.cat([states, actions], dim=-1)
    # output = torch.cos(self.fourier(obs_action))
    # print("states", states)
    next_states = self.dynamics_fn(states, self.rescale_action(actions))
    # print("next_states", next_states)
    output = torch.cos(self.fourier(next_states))
    # print("output", output)
    return output


class randPhi_s_prime(nn.Module):
  """phi(s') random function"""
  def __init__(self, obs_dim, output_dim, dynamics_fn = None, action_range = None,sigma = 1.,device = 'cpu'):
    super().__init__()
    # fourier_feats = nn.Linear(obs_dim + action_dim, output_dim)
    fourier_feats = nn.Linear(obs_dim, output_dim)
    init.normal_(fourier_feats.weight, std = 1./sigma)
    init.uniform_(fourier_feats.bias, 0, 2 * np.pi)
    fourier_feats.weight.requires_grad = False
    fourier_feats.bias.requires_grad = False
    self.fourier = fourier_feats



  def forward(self, states:torch.Tensor):
    # obs_action = torch.cat([states, actions], dim=-1)
    # output = torch.cos(self.fourier(obs_action))
    # print("states", states)
    # print("next_states", next_states)
    output = torch.cos(self.fourier(states))
    # print("output", output)
    return output

class nnPhi_s_prime(nn.Module):
  """phi(s') nn network"""
  def __init__(self, obs_dim, output_dim, dynamics_fn = None, action_range = None,sigma = 1.,device = 'cpu'):
    super().__init__()
    self.nn = util.mlp(obs_dim, output_dim = output_dim, hidden_dim =  256, hidden_depth = 2)



  def forward(self, states:torch.Tensor):
    # obs_action = torch.cat([states, actions], dim=-1)
    # output = torch.cos(self.fourier(obs_action))
    # print("states", states)
    # print("next_states", next_states)
    output =self.nn(states)
    # print("output", output)
    return output