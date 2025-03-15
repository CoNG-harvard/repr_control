"""
We need to define the nonlinear control problems in this file.
Problem instanced based on Kuramoto model
"""

import torch
import numpy as np

### This is a networked problem ###

torch.manual_seed(0)


########################################################################################################################
# 1. define problem-related constants
########################################################################################################################
state_dim = 2                    # state dimension
action_dim = 1                      # action dimension
N = 10 # num of agents
# state_range = [[15] * N,
#                [40] * N]           # low and high. We set bound on the state to ensure stable training.
state_range = [[-1,-1] * N,
               [1,1] * N]           # low and high. We set bound on the state to ensure stable training.
action_range = [[-4] * N, [4] * N]          # low and high
max_step = 500                      # maximum rollout steps per episode
sigma = 0.05                      # noise standard deviation.
env_name = 'kuramoto'
assert len(action_range[0]) == len(action_range[1]) == action_dim * N


curr_device = 'cuda:0'



def build_adjacency(N):
    adjacency = torch.zeros((N,N),dtype = torch.int)
    for i in range(N):
        if 1 <= i <= N-2:
            i_lower = i-1
            i_upper = i + 1
        elif i == 0:
            i_lower = N-1
            i_upper = 1 
        elif i == N-1:
            i_lower = N-2
            i_upper = 0
        i_lower = max(i_lower,0)
        i_upper = min(i_upper, N-1)
        adjacency[i,i_lower] = 1
        adjacency[i,i] = 1
        adjacency[i,i_upper] = 1
    return adjacency

##construct the (symmetric) P matrix for use later in the dynamics.
def get_P(N, adjacency, seed = 0):
    P = torch.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if adjacency[i,j] == 1 and i != j: 
                rand_val = torch.rand(1) + 0.2
                P[i,j] = rand_val
                P[j,i] = rand_val
    return P.to(curr_device)

def get_neighbors(N,adjacency):
    neighbors = [[] for i in range(N)]
    for i in range(N):
        for j in range(N):
            if adjacency[i,j] == 1:
                neighbors[i] += [j]
    return torch.tensor(neighbors, dtype = torch.int)


omega = torch.rand(size = (N,), device=curr_device) * 2.0

adjacency = build_adjacency(N)
# adjacency = torch.ones((N,N), dtype = torch.int) #try everybody connected for now.
P = get_P(N,adjacency)
print("P", P)
print("adjacency")
print("omega", omega)



policy_adjacency = get_neighbors(N, adjacency)
eval_adjacency = get_neighbors(N, adjacency)
print("policy_adjacency", policy_adjacency)
kappa_obs_dim = 6 #this is the dimension of the concatenation of the states of an agent's kappa-neighborhood neighbors
eval_kappa_obs_dim = 6
eval_kappa_action_dim = 3



########################################################################################################################
# 2. define dynamics model, reward function and initial distribution.
########################################################################################################################
def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    The dynamics. Needs to be written in pytorch to enable auto differentiation.
    The input and outputs should be 2D Tensors, where the first dimension should be batch size, and the second dimension 
    is the state. For Local kuramoto, states are 
    [[cos(th1), sin(th1)],
    ...

    [cos(thN), sin(thN)]]
    
    Parameters
    ----------
    state            torch.Tensor, [batch_size, state_dim * N ] 
    action           torch.Tensor, [batch_size, action_dim * N]

    Returns
    next_state       torch.Tensor, [batch_size, state_dim*N]
    -------

    """

    # dt = 0.05
    dt = 0.01

    cos_th, sin_th = state[:,::2], state[:,1::2]
    th = torch.atan2(sin_th, cos_th).reshape((-1, N))

    th_expanded = th.unsqueeze(2)



    th_diff = th_expanded - th_expanded.transpose(1,2)

    sin_th_diff = torch.sin(th_diff)



    thdot = omega - torch.sum(sin_th_diff * P, axis = -1) + action

    newth = th + dt * thdot

    cos_newth = torch.cos(newth).unsqueeze(2)
    sin_newth = torch.sin(newth).unsqueeze(2)
    interleaved = torch.cat((cos_newth, sin_newth), dim=2)
    next_state = interleaved.view((state.size(0), -1))

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
    cos_th, sin_th = state[:,::2], state[:,1::2]
    th = torch.atan2(sin_th, cos_th).reshape((-1, N))

    th_expanded = th.unsqueeze(2)
    th_diff_squared = (th_expanded - th_expanded.transpose(1,2))**2


    local_diff_sq = (th_diff_squared * P).sum(dim = 2)

    # reward = -1 * (local_diff_sq  + 0.01 * action ** 2)
    reward = -1 * local_diff_sq
    return reward

# output is tensor of dimension (batch_size, N)
def initial_distribution(batch_size: int) -> torch.Tensor:
    th = 2 * np.pi * torch.rand((batch_size,N)) - np.pi 
    cos_th = torch.cos(th).unsqueeze(2)
    sin_th = torch.sin(th).unsqueeze(2)
    interleaved = torch.cat((cos_th, sin_th), dim=2)
    init_state = interleaved.view((batch_size, -1))
    return init_state
