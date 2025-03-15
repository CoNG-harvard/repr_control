import torch
import numpy as np

########################################################################################################################
# 1. define problem-related constants
########################################################################################################################
state_dim = 5  # Updated state dimension to 5
action_dim = 1
state_range = [[-2.4 * 2, -np.finfo(np.float32).max, -1.0, -1.0, -np.finfo(np.float32).max],  # low bounds for cos(theta), sin(theta)
               [2.4 * 2, np.finfo(np.float32).max, 1.0, 1.0, np.finfo(np.float32).max]]      # high bounds for cos(theta), sin(theta)
action_range = [[-1], [1]]
max_step = 200
env_name = 'ContinuousCartPole'
sigma = 0.05
assert len(action_range[0]) == len(action_range[1]) == action_dim

########################################################################################################################
# 2. define dynamics model, reward function and initial distribution.
########################################################################################################################
def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Dynamics function with theta as cos(theta) and sin(theta).
    """
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole * length
    force_mag = 30.0
    tau = 0.02
    max_action = 1.0

    x, x_dot, costheta, sintheta, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]
    theta = torch.atan2(sintheta, costheta)

    action = torch.reshape(action, (action.shape[0],))
    force = force_mag * torch.clip(action, -max_action, max_action)

    temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta_dot = theta_dot + tau * thetaacc
    theta = theta + tau * theta_dot
    costheta, sintheta = torch.cos(theta), torch.sin(theta)

    next_state = torch.vstack([x, x_dot, costheta, sintheta, theta_dot]).T
    assert next_state.shape[1] == state_dim  # Check the updated state shape
    return next_state

def rewards(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Reward function based on cos(theta).
    """
    x, x_dot, costheta, sintheta, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]
    reward = costheta - 0.01 * x**2
    return reward

def initial_distribution(batch_size: int) -> torch.Tensor:
    """
    Initializes states with cos(theta) and sin(theta).
    """
    x = 0.1 * (2 * torch.rand((batch_size)) - 1)
    x_dot = 0.1 * (2 * torch.rand((batch_size)) - 1)
    theta = 0.1 * (2 * torch.rand((batch_size)) - 1)
    theta_dot = 0.1 * (2 * torch.rand((batch_size)) - 1)
    costheta, sintheta = torch.cos(theta), torch.sin(theta)
    return torch.vstack([x, x_dot, costheta, sintheta, theta_dot]).T

def rand_distribution(batch_size: int) -> torch.Tensor:
    """
    Generates random states with cos(theta) and sin(theta).
    """
    x = 2.4 * (2 * torch.rand((batch_size)) - 1)
    x_dot = 2 * torch.rand((batch_size)) - 1
    theta = 12 * 2 * np.pi / 360 * (2 * torch.rand((batch_size)) - 1)
    theta_dot = 2 * torch.rand((batch_size)) - 1
    costheta, sintheta = torch.cos(theta), torch.sin(theta)
    return torch.vstack([x, x_dot, costheta, sintheta, theta_dot]).T
