"""
We adapt the code from https://github.com/denisyarats/pytorch_sac
"""


import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

from torch import vmap

from repr_control.utils import util_network


class TanhTransform(pyd.transforms.Transform):
  domain = pyd.constraints.real
  codomain = pyd.constraints.interval(-1.0, 1.0)
  bijective = True
  sign = +1

  def __init__(self, cache_size=1):
    super().__init__(cache_size=cache_size)

  @staticmethod
  def atanh(x):
    return 0.5 * (x.log1p() - (-x).log1p())

  def __eq__(self, other):
    return isinstance(other, TanhTransform)

  def _call(self, x):
    return x.tanh()

  def _inverse(self, y):
    # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
    # one should use `cache_size=1` instead
    return self.atanh(y)

  def log_abs_det_jacobian(self, x, y):
    # We use a formula that is more numerically stable, see details in the following link
    # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
    return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

    self.base_dist = pyd.Normal(loc, scale, validate_args=False)
    transforms = [TanhTransform()]
    super().__init__(self.base_dist, transforms, validate_args = False)

  @property
  def mean(self):
    mu = self.loc
    for tr in self.transforms:
        mu = tr(mu)
    return mu


# class DiagGaussianActor(nn.Module):
#   """torch.distributions implementation of an diagonal Gaussian policy."""
#   def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
#                 log_std_bounds):
#     super().__init__()
#     # print("obs_dim: ",obs_dim)

#     self.log_std_bounds = log_std_bounds
#     self.trunk = util.mlp(obs_dim, hidden_dim, 2 * action_dim,
#                             hidden_depth)

#     self.outputs = dict()
#     self.apply(util.weight_init)

#   def forward(self, obs):
#     mu, log_std = self.trunk(obs).chunk(2, dim=-1)

#     # constrain log_std inside [log_std_min, log_std_max]
#     log_std = torch.tanh(log_std)
#     log_std_min, log_std_max = self.log_std_bounds
#     log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
#                                                                   1)

#     std = log_std.exp()

#     self.outputs['mu'] = mu
#     self.outputs['std'] = std

#     dist = SquashedNormal(mu, std)
#     return dist

class DiagGaussianActor(nn.Module):
  """torch.distributions implementation of an diagonal Gaussian policy."""
  def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                log_std_bounds):
    super().__init__()
    # print("obs_dim: ",obs_dim)

    self.log_std_bounds = log_std_bounds
    self.trunk = util_network.mlp(obs_dim, hidden_dim, 2 * action_dim,
                            hidden_depth)

    self.outputs = dict()
    self.apply(util_network.weight_init)

  def forward(self, obs):
    mu, log_std = self.trunk(obs).chunk(2, dim=-1)


    # print("this is std", std)
    # print("this is mu before", mu)
    # print("this is log_std before", log_std)

    # constrain log_std inside [log_std_min, log_std_max]
    log_std = torch.tanh(log_std)
    log_std_min, log_std_max = self.log_std_bounds
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                  1)

    std = log_std.exp()

    self.outputs['mu'] = mu
    self.outputs['std'] = std

    # print("this is std", std)
    # print("this is log_std", log_std)

    # dist = SquashedNormal(mu, std)
    return mu, std




class NetworkDiagGaussianActor(nn.Module):
  """torch.distributions implementation of a networked diagonal Gaussian policy."""
  def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                log_std_bounds):
    super().__init__()
    # print("obs_dim: ",obs_dim)

    self.log_std_bounds = log_std_bounds
    self.trunk_layers = nn.ParameterDict()
    self.init_trunk_layers()

    # self.trunk = util.mlp(obs_dim, hidden_dim, 2 * action_dim,
    #                         hidden_depth)

    self.outputs = dict()
    self.apply(util_network.weight_init)

  def forward(self, obs):
    mu, log_std = self.trunk(obs).chunk(2, dim=-1)

    # constrain log_std inside [log_std_min, log_std_max]
    log_std = torch.tanh(log_std)
    log_std_min, log_std_max = self.log_std_bounds
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

    std = log_std.exp()

    self.outputs['mu'] = mu
    self.outputs['std'] = std

    dist = SquashedNormal(mu, std)
    return dist

  def init_trunk_layers(self,N,input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
      self.trunk_layers["layer0"] = nn.Parameter((torch.rand(size = (N, input_dim, output_dim)) - 0.5 ) * 2 * 1./output_dim)
    if hidden_depth > 0:
      self.trunk_layers["layer0"] = nn.Parameter((torch.rand(size = (N, input_dim, hidden_dim)) - 0.5) * 2./hidden_dim)
      for i in range(0,hidden_depth-1):
        self.trunk_layers["hidden-layer%d"%(i)] = nn.Parameter((torch.rand(size = (N, hidden_dim, hidden_dim)) - 0.5) * 2./hidden_dim)
      self.trunk_layers["finallayer"] = nn.Parameter((torch.rand(size = (N, hidden_dim, output_dim)) - 0.5) * 2./output_dim)



class DeterministicActor(nn.Module):
  def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
    super().__init__()
    self.trunk = util.mlp(obs_dim, hidden_dim, action_dim,
                            hidden_depth)

    self.outputs = dict()
    self.apply(util.weight_init)

  def forward(self, obs):
    action = self.trunk(obs).tanh()
    return action


class StochasticActorFromDetStructureWrapper(nn.Module):
  """
  It's a simple wrapper that wraps stochastic policy to
  """
  def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds, det_module):
    super().__init__()
    self.log_std_bounds = log_std_bounds
    self.trunk = util_network.mlp(obs_dim, hidden_dim, action_dim,
                          hidden_depth)

    self.outputs = dict()
    self.apply(util_network.weight_init)
    self.det_module = det_module
    self.action_dim = action_dim

  def forward(self, obs):
    mu = self.det_module(obs)[:, :self.action_dim]

    log_std = self.trunk(obs)
    assert mu.shape == log_std.shape, "mu std shape not consistent"

    # constrain log_std inside [log_std_min, log_std_max]
    log_std = torch.tanh(log_std)
    log_std_min, log_std_max = self.log_std_bounds
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                  1)

    std = log_std.exp()

    self.outputs['mu'] = mu
    self.outputs['std'] = std

    dist = SquashedNormal(mu, std)
    return dist