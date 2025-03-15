import torch
import numpy as np

########################################################################################################################
# 1. Define problem-related constants
########################################################################################################################
state_dim = 2                       # state dimension (position, velocity)
action_dim = 1                      # action dimension (force)
state_range = [[-1.2, -0.07],       # low bounds
               [0.6, 0.07]]         # high bounds
action_range = [[-1], [1]]          # low and high action bounds
max_step = 999                      # maximum rollout steps per episode
env_name = 'ContinuousMountainCar'
sigma = 0.05
goal_position = 0.45
goal_velocity = 0.0
assert len(action_range[0]) == len(action_range[1]) == action_dim

########################################################################################################################
# 2. Define dynamics model, reward function, and initial distribution
########################################################################################################################
def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    The dynamics. Written in PyTorch to enable auto differentiation.
    
    Parameters
    ----------
    state            torch.Tensor, [batch_size, state_dim]
    action           torch.Tensor, [batch_size, action_dim]

    Returns
    -------
    next_state       torch.Tensor, [batch_size, state_dim]
    """
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    power = 0.0015

    position, velocity = state[:, 0], state[:, 1]
    force = torch.clamp(action[:, 0], min=action_range[0][0], max=action_range[1][0])

    # Update dynamics
    velocity = velocity + force * power - 0.0025 * torch.cos(3 * position)
    velocity = torch.clamp(velocity, min=-max_speed, max=max_speed)

    position = position + velocity
    position = torch.clamp(position, min=min_position, max=max_position)

    # Set velocity to 0 if at boundary
    velocity = torch.where((position == min_position) & (velocity < 0), 0, velocity)

    next_state = torch.stack([position, velocity], dim=1)
    return next_state

def rewards(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    The reward. Written in PyTorch to enable auto differentiation.

    Parameters
    ----------
    state            torch.Tensor, [batch_size, state_dim]
    action           torch.Tensor, [batch_size, action_dim]

    Returns
    -------
    rewards          torch.Tensor, [batch_size,]
    """
    position, velocity = state[:, 0], state[:, 1]

    # Compute termination condition
    done = (position >= goal_position) & (velocity >= goal_velocity)

    # Reward calculation
    reward = torch.where(done, 100.0, 0.0)
    reward -= 0.1 * action[:, 0]**2  # Penalize large actions

    return reward

def initial_distribution(batch_size: int) -> torch.Tensor:
    """
    Generate initial state distribution.

    Parameters
    ----------
    batch_size       int, batch size

    Returns
    -------
    init_state       torch.Tensor, [batch_size, state_dim]
    """
    position = -0.6 + 0.2 * torch.rand(batch_size)
    velocity = torch.zeros(batch_size)
    return torch.stack([position, velocity], dim=1)

def rand_distribution(batch_size: int) -> torch.Tensor:
    """
    Generate random state distribution for testing.

    Parameters
    ----------
    batch_size       int, batch size

    Returns
    -------
    rand_state       torch.Tensor, [batch_size, state_dim]
    """
    position = (1.8 * torch.rand(batch_size)) - 1.2
    velocity = (0.14 * torch.rand(batch_size)) - 0.07
    return torch.stack([position, velocity], dim=1)
