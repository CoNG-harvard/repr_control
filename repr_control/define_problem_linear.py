"""
We need to define the nonlinear control problems in this file.
"""

import torch
import numpy as np
########################################################################################################################
# 1. define problem-related constants
########################################################################################################################
state_dim = 1                  # state dimension
action_dim = 1                      # action dimension
state_range = [[-2],
               [2]]           # low and high. We set bound on the state to ensure stable training.
action_range = [[-2], [2]]          # low and high
max_step = 200                      # maximum rollout steps per episode
sigma = 0.5                # noise standard deviation.
env_name = 'Linear'
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
    next_state = 0.5 * state
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
    action = torch.reshape(action, (action.shape[0],))
    reward = -0.3 * (state[:,0] ** 2 +  0.001 * action ** 2)
    return reward

def initial_distribution(batch_size: int) -> torch.Tensor:
    init = 2* (torch.rand((batch_size,state_dim)) - 0.5)
    return init


def rand_distribution(batch_size: int) -> torch.Tensor:
    init = 2* (torch.rand((batch_size,state_dim)) - 0.5) * 2
    return init
