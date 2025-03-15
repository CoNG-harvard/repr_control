"""
We need to define the nonlinear control problems in this file.
"""

import torch
import numpy as np

### This is a networked problem ###


########################################################################################################################
# 1. define problem-related constants
########################################################################################################################
state_dim = 3                     # state dimension
action_dim = 1                      # action dimension
N = 1 # num of agents
state_range = [[-1, -1, -8],
               [1, 1, 8]]           # low and high. We set bound on the state to ensure stable training.
action_range = [[-2], [2]]          # low and high
max_step = 200                      # maximum rollout steps per episode
sigma = 0.05                          # noise standard deviation.
env_name = 'pendulum_disjoint'
assert len(action_range[0]) == len(action_range[1]) == action_dim * N

curr_device = 'cuda:0'





adjacency = torch.reshape(torch.tensor([0],dtype = torch.int), (N,1))
print("adjacency", adjacency)

policy_adjacency = adjacency
eval_adjacency = adjacency
kappa_obs_dim = 3 #this is the dimension of the concatenation of the states of an agent's kappa-neighborhood neighbors
eval_kappa_obs_dim = 3



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
    state            torch.Tensor, [batch_size, state_dim * N ] 
    action           torch.Tensor, [batch_size, action_dim * N]

    Returns
    next_state       torch.Tensor, [batch_size, state_dim*N]
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
    action1 = torch.reshape(action[:,0], (action.shape[0],))
    u = torch.clip(action1, -max_a, max_a)
    newthdot = thdot + (3. * g / (2 * l) * torch.sin(th) + 3.0 / (m * l ** 2) * u) * dt
    newthdot = torch.clip(newthdot, -max_speed, max_speed)
    newth = th + newthdot * dt
    next_state = torch.vstack([torch.cos(newth), torch.sin(newth), newthdot]).T

    # cos_th2, sin_th2, thdot2 = state[:, 3], state[:, 4], state[:, 5]  
    # th2 = torch.atan2(sin_th2, cos_th2)
    # action2 = torch.reshape(action[:,1], (action.shape[0],))
    # u2 = torch.clip(action2, -max_a, max_a)
    # newthdot2 = thdot2 + (3. * g / (2 * l) * torch.sin(th2) + 3.0 / (m * l ** 2) * u2) * dt
    # newthdot2 = torch.clip(newthdot2, -max_speed, max_speed)
    # newth2 = th2 + newthdot2 * dt
    # next_state2 = torch.vstack([torch.cos(newth2), torch.sin(newth2), newthdot2]).T

    # next_state = torch.hstack([next_state,next_state2])

    assert next_state.shape == state.shape
    return next_state

def rewards(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    The reward. Needs to be written in pytorch to enable auto differentiation.
    
    Parameters
    ----------
    state            torch.Tensor, [batch_size, state_dim * N] 
    action           torch.Tensor, [batch_size, action_dim * N]

    Returns
    rewards       torch.Tensor, [batch_size,N]
    -------

    """
    cos_th, sin_th, thdot = state[:, 0], state[:, 1], state[:, 2]
    th = torch.atan2(sin_th, cos_th)
    action1 = torch.reshape(action[:,0], (action.shape[0],))
    reward = -0.3 * (th ** 2 + 0.1 * thdot ** 2 + 0.001 * action1 ** 2)

    # cos_th2, sin_th2, thdot2 = state[:, 3], state[:, 4], state[:, 5]
    # th2 = torch.atan2(sin_th2, cos_th2)
    # action2 = torch.reshape(action[:,1], (action.shape[0],))
    # reward2 = -0.3 * (th2 ** 2 + 0.1 * thdot2 ** 2 + 0.001 * action2 ** 2)

    reward1 = torch.reshape(reward,(reward.shape[0],-1))
    # reward2 = torch.reshape(reward2,(reward.shape[0],-1))
    # reward = torch.hstack([reward1,reward2])
    # print("reward shape", reward.shape)

    # action = torch.reshape(action, (action.shape[0],))
    # reward = -1 * ((T_set - state) ** 2 + 0.1 * action ** 2)
    return reward1

# output is tensor of dimension (batch_size, N)
def initial_distribution(batch_size: int) -> torch.Tensor:
    th = 2 * np.pi * torch.rand((batch_size)) - np.pi
    thdot = 2 * torch.rand((batch_size)) - 1

    init1 = torch.vstack([torch.cos(th),
                         torch.sin(th),
                         thdot]).T

    # th2 = 2 * np.pi * torch.rand((batch_size)) - np.pi
    # thdot2 = 2 * torch.rand((batch_size)) - 1
    # init2 = torch.vstack([torch.cos(th2),
    #                      torch.sin(th2),
    #                      thdot2]).T
    # init = torch.hstack([init1,init2])
    return init1
