"""
We need to define the nonlinear control problems in this file.
"""

import torch
import numpy as np
########################################################################################################################
# 1. define problem-related constants
########################################################################################################################
state_dim = 3   
obs_dim = 3                    # state dimension
action_dim = 1                      # action dimension
state_range = [[-1, -1, -8],
               [1, 1, 8]]           # low and high. We set bound on the state to ensure stable training.
action_range = [[-2], [2]]          # low and high
# max_step = 200                      # maximum rollout steps per episode
max_step = 500
sigma = 0.05                   # noise standard deviation.
env_name = 'Pendulum'
assert len(action_range[0]) == len(action_range[1]) == action_dim

########################################################################################################################
# 2. define dynamics model, reward function and initial distribution.
########################################################################################################################
def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    The dynamics. Needs to be written in pytorch to enable auto differentiation.
    The input and outputs should be 2D Tensors, where the first dimension should be batch size, and the second dimension 
    is the state. For example, the pendulum state will looks like
    [[cos(theta), sin(theta), dot theta],
     [cos(theta), sin(theta), dot theta],
     ...,
     [cos(theta), sin(theta), dot theta]
     ]
    
    Parameters
    ----------
    state            torch.Tensor, [batch_size, state_dim] 
    action           torch.Tensor, [batch_size, action_dim]

    Returns
    next_state       torch.Tensor, [batch_size, state_dim]
    -------

    """
    g = 10.0
    m = 1.
    l = 1.
    max_a = 2.
    dt = 0.05
    max_speed = 8
    cos_th, sin_th, thdot = state[:, 0], state[:, 1], state[:, 2]
    th = torch.atan2(sin_th, cos_th)
    action = torch.reshape(action, (action.shape[0],))
    u = torch.clip(action, -max_a, max_a)
    newthdot = thdot + (3. * g / (2 * l) * torch.sin(th) + 3.0 / (m * l ** 2) * u) * dt
    newthdot = torch.clip(newthdot, -max_speed, max_speed)
    newth = th + newthdot * dt
    next_state = torch.vstack([torch.cos(newth), torch.sin(newth), newthdot]).T
    assert next_state.shape == state.shape
    return next_state

def rewards(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    The reward. Needs to be written in pytorch to enable auto differentiation.
    
    Parameters
    ----------
    state            torch.Tensor, [batch_size, state_dim] 
    action           torch.Tensor, [batch_size, action_dim]

    Returns
    rewards       torch.Tensor, [batch_size,]
    -------

    """
    cos_th, sin_th, thdot = state[:, 0], state[:, 1], state[:, 2]
    th = torch.atan2(sin_th, cos_th)
    action = torch.reshape(action, (action.shape[0],))
    reward = -0.3 * (th ** 2 + 0.1 * thdot ** 2 + 0.001 * action ** 2)
    return reward

def initial_distribution(batch_size: int) -> torch.Tensor:
    th = 2 * np.pi * torch.rand((batch_size)) - np.pi
    thdot = 2 * torch.rand((batch_size)) - 1
    return torch.vstack([torch.cos(th),
                         torch.sin(th),
                         thdot]).T


def rand_distribution(batch_size: int) -> torch.Tensor:
    th = (2 * np.pi * torch.rand((batch_size)) - np.pi) * 1
    thdot = (2 * torch.rand((batch_size)) - 1) * 8
    return torch.vstack([torch.cos(th),
                         torch.sin(th),
                         thdot]).T
