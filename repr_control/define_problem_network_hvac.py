"""
We need to define the nonlinear control problems in this file.
Problem instanced based on "Distributed Model Predictive Control of 
Bilinear HVAC Systems Using a Convexification Method"
"""

import torch
import numpy as np

### This is a networked problem ###

##specify device

curr_device = 'cuda:1'

########################################################################################################################
# 1. define problem-related constants
########################################################################################################################
state_dim = 1                     # state dimension
action_dim = 1                      # action dimension
# N = 3 # num of agents
N = 1
kappa = 0
state_range = [[-5] * N,
               [5] * N]           # low and high. We set bound on the state to ensure stable training.
action_range = [[0] * N, [4] * N]          # low and high
# max_step = 200                      # maximum rollout steps per episode
max_step = 50
sigma = 0.01                        # noise standard deviation.
env_name = 'nonlinear_HVAC'
assert len(action_range[0]) == len(action_range[1]) == action_dim * N

T_set = 25



thermal_resistance_zones = torch.ones((N,N), device = curr_device) * 14
thermal_resistance_outside = torch.ones(N, device = curr_device) * 50

def build_adjacency(N):
    adjacency = torch.zeros((N,N),dtype = torch.int, device = curr_device)
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
        i_upper = min(i_upper,N-1) 
        i_lower = max(i_lower,0) 
        adjacency[i,i_lower] = 1
        adjacency[i,i] = 1
        adjacency[i,i_upper] = 1
    return adjacency

##construct the (symmetric) A matrix for use later in the dynamics.
def get_A(N, adjacency,thermal_resistance_zones,thermal_resistance_outside):
    A = torch.zeros((N,N), device = curr_device )
    for i in range(N):
        A[i,i] = -1./thermal_resistance_zones[i,i] - 1./thermal_resistance_outside[i]
        for j in range(N):
            if adjacency[i,j] == 1 and i != j: 
                A[i,j] = 1./thermal_resistance_zones[i,j]
    return A

def get_neighbors(N,adjacency):
    neighbors = [[] for i in range(N)]
    for i in range(N):
        for j in range(N):
            if adjacency[i,j] == 1:
                neighbors[i] += [j]
    return torch.tensor(neighbors, dtype = torch.int, device = curr_device)


adjacency = build_adjacency(N)
A = get_A(N,adjacency, thermal_resistance_zones,thermal_resistance_outside)
print("A", A)
print("adjacency")



policy_adjacency = get_neighbors(N, adjacency)
eval_adjacency = get_neighbors(N, adjacency)
print("policy_adjacency", policy_adjacency)
kappa_obs_dim =  2 * kappa +1 #this is the dimension of the concatenation of the states of an agent's kappa-neighborhood neighbors
eval_kappa_obs_dim = 2 * kappa +1



########################################################################################################################
# 2. define dynamics model, reward function and initial distribution.
########################################################################################################################
def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    The dynamics. Needs to be written in pytorch to enable auto differentiation.
    The input and outputs should be 2D Tensors, where the first dimension should be batch size, and the second dimension 
    is the state. 

    State = temp - 25
    
    Parameters
    ----------
    state            torch.Tensor, [batch_size, state_dim * N ] 
    action           torch.Tensor, [batch_size, action_dim * N]

    Returns
    next_state       torch.Tensor, [batch_size, state_dim*N]
    -------

    """

    temp = state + T_set

    cap = torch.ones(N, device= curr_device) * (1.35 * 1e3)
    air_cap = 1.012
    T_supp = torch.ones(N, device = curr_device) * 15
    # T_supp = torch.ones(N, device = curr_device) * 20
    T_outside = 32
    dt = 10 #min

    temp_grad = torch.matmul(temp, A) + T_outside/thermal_resistance_outside + action * air_cap * (T_supp - temp)
    # state_grad = state_grad

    next_temp = temp + dt/cap * temp_grad

    next_state = next_temp - T_set
    # print("state", state)
    # print("next state", next_state)
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
    # T_set = torch.ones(N, device= curr_device) * 25




    # action = torch.reshape(action, (action.shape[0],))
    # print("action shape", action.shape)
    # print("torch abs diff shape", torch.abs(T_set - state).shape)
    # reward = -1 * (torch.abs(state)**3  + 0.001 * action ** 2)
    # reward = -1 * (torch.abs(state)  + 0.01 * action ** 2)
    reward = -1 * ((T_set - state)**2  + 0.01 * action ** 2)
    return reward

# output is tensor of dimension (batch_size, N)
def initial_distribution(batch_size: int) -> torch.Tensor:
    # temp = torch.rand(size = (batch_size,N), device = curr_device) * 5 + 26
    temp = torch.rand(size = (batch_size,N), device = curr_device) * 5
    # temp = torch.rand(size = (batch_size,N), device = curr_device) * -5
    # temp = (torch.rand(size = (batch_size,N), device = curr_device) - 0.5) * 10
    return temp
