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
state_dim = 2                 # state dimension
action_dim = 1                      # action dimension
N = 10 # num of agents
policy_kappa = 1
eval_kappa = 1
# state_range = [[15] * N,
#                [40] * N]           # low and high. We set bound on the state to ensure stable training.
state_range = [[-1,-1] * N,
               [1,1] * N]           # low and high. We set bound on the state to ensure stable training.
# action_range = [[-3] * N, [3] * N]          # low and high
action_range = [[-3] * N, [3] * N]          # low and high
max_step = 800                      # maximum rollout steps per episode
sigma = 0.01                  # noise standard deviation.
env_name = 'kuramoto_w_kappa_thdot_2nd_order'
assert len(action_range[0]) == len(action_range[1]) == action_dim * N


curr_device = 'cuda:0'



def build_adjacency(N,kappa):
    adjacency = torch.zeros((N,N),dtype = torch.int)
    for i in range(N):
        adjacency[i,i] = 1
        for j in range(kappa+1):
            j_minus_idx = i - j
            if i + j <= N- 1:
                j_plus_idx = i + j
            else:
                j_plus_idx = i + j - N
            adjacency[i,j_plus_idx] = 1
            adjacency[i,j_minus_idx] = 1
    return adjacency

##construct the (symmetric) P matrix for use later in the dynamics.
def get_P(N, adjacency, seed = 0, scale = 1.):
    P = torch.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if adjacency[i,j] == 1 and i != j: 
                rand_val = scale * (torch.rand(1) + 0.2)
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


# omega = torch.rand(size = (N,), device=curr_device) * 2.0


# omega = torch.rand(size = (N,), device=curr_device) * 1.5

omega = torch.rand(size = (N,), device=curr_device) * 1.5



adjacency = build_adjacency(N,kappa = policy_kappa).to(curr_device)
kappa_minus_one_adjacency = build_adjacency(N,kappa = eval_kappa-1)
eval_adjacency = build_adjacency(N,kappa = eval_kappa)
# adjacency = torch.ones((N,N), dtype = torch.int) #try everybody connected for now.
P = get_P(N,adjacency, scale = 1.)
# P = get_P(N,adjacency, scale = 2.)
print("P", P)
print("adjacency", adjacency)
print("omega", omega)



policy_adjacency = get_neighbors(N, adjacency)
eval_minus_one_adjacency = get_neighbors(N, kappa_minus_one_adjacency)
eval_adjacency = get_neighbors(N, eval_adjacency)
print("policy_adjacency", policy_adjacency)
kappa_obs_dim = (2 * policy_kappa + 1) * state_dim #this is the dimension of the concatenation of the states of an agent's kappa-neighborhood neighbors
eval_kappa_obs_dim = (2*eval_kappa + 1) * state_dim
eval_kappa_action_dim = (2*eval_kappa + 1) * action_dim



########################################################################################################################
# 2. define dynamics model, reward function and initial distribution.
########################################################################################################################
def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    dt = 0.05  # Time step
    M = 1.0    # Inertia coefficient
    D = 0.1    # Damping coefficient

    batch_size = state.shape[0]
    N = state.shape[1] // 2

    # Extract theta_i and dot_theta_i
    theta = state[:, 0::2]       # Shape: [batch_size, N]
    dot_theta = state[:, 1::2]   # Shape: [batch_size, N]

    # Compute pairwise differences and interactions
    theta_diff = theta.unsqueeze(2) - theta.unsqueeze(1)  # Shape: [batch_size, N, N]
    sin_theta_diff = torch.sin(theta_diff)

    # Compute coupling term
    coupling = torch.sum(P * sin_theta_diff, dim=2)       # Shape: [batch_size, N]

    # Compute acceleration \ddot{\theta}_i
    ddot_theta = (1.0 / M) * (omega - D * dot_theta - coupling + action)  # Shape: [batch_size, N]

    # Update dot_theta_i
    new_dot_theta = dot_theta + dt * ddot_theta

    # Update theta_i
    new_theta = theta + dt * new_dot_theta

    # Wrap new_theta to [-π, π]
    new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi

    # Construct next_state
    next_state = torch.zeros_like(state)
    next_state[:, 0::2] = new_theta
    next_state[:, 1::2] = new_dot_theta

    return next_state


def rewards(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    theta = state[:, 0::2]
    dot_theta = state[:, 1::2]

    dot_theta_diff = dot_theta.unsqueeze(2) - dot_theta.unsqueeze(1)
    velocity_sync_error = torch.sum((dot_theta_diff ** 2) * adjacency, dim=2)

    # Optionally, include phase synchronization
    theta_diff = theta.unsqueeze(2) - theta.unsqueeze(1)
    phase_sync_error = torch.sum((torch.sin(theta_diff / 2) ** 2) * adjacency, dim=2)

    reward = - (velocity_sync_error + phase_sync_error)
    # print("reward shape", reward.shape)

    # Optionally, penalize control effort
    # reward -= 0.01 * torch.sum(action ** 2, dim=1)

    return reward

# output is tensor of dimension (batch_size, N)
def initial_distribution(batch_size: int) -> torch.Tensor:
    # N = state_dim * batch_size // 2
    theta = 2 * np.pi * torch.rand((batch_size, N)) - np.pi  # Random theta_i in [-π, π]
    dot_theta = torch.zeros((batch_size, N))  # Initialize velocities to zero or small random values

    # Optionally, initialize dot_theta with small random values
    # dot_theta = 0.1 * torch.randn((batch_size, N))

    # Construct initial state
    init_state = torch.zeros((batch_size, 2 * N))
    init_state[:, 0::2] = theta
    init_state[:, 1::2] = dot_theta

    return init_state

