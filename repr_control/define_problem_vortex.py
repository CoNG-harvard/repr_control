import torch
import numpy as np
import math

########################################################################################################################
# 1. Define Problem-Related Constants
########################################################################################################################
state_dim = 8  # State dimension: x (4-dim) and fx (4-dim)
action_dim = 2  # Action dimension
state_range = [[-0.05] * state_dim,   # Lower bounds for each state component
               [0.05] * state_dim]     # Upper bounds for each state component
action_range = [[-1.0, -1.0], [1.0, 1.0]]  # Action bounds
max_step = 800  # Maximum rollout steps per episode
sigma = 0.05  # Noise standard deviation
env_name = 'Vortex'

# Physical and Control Constants
lmbda_re = 9.153
lmbda_cx = 3.239
mu_re = 308.9
mu_cx = -1025.0
alpha_re = 0.03492
alpha_cx = 0.01472
beta = 1.0

re = 50.0
re_crit = 46.6
ire = 1.0 / re_crit - 1.0 / re

omega_s = 1.1
omega_f = 0.74
domega = omega_s - omega_f

gamma = 0.023
mass = 10.0
weight = 50.0
beta_m = beta / (omega_f * mass)

dt = 0.1   # Numerical timestep
dt_act = 0.5  # Action timestep

mod_min = 0.0
mod_max = 0.3
phase_min = -math.pi
phase_max = math.pi

# LSRK4 Coefficients
lsrk4_a = torch.tensor([0.000000000000000, -0.417890474499852,
                        -1.192151694642677, -1.697784692471528,
                        -1.514183444257156], dtype=torch.float32)
lsrk4_b = torch.tensor([0.149659021999229,  0.379210312999627,
                        0.822955029386982,  0.699450455949122,
                        0.153057247968152], dtype=torch.float32)
lsrk4_c = torch.tensor([0.000000000000000,  0.149659021999229,
                        0.370400957364205,  0.622255763134443,
                        0.958282130674690], dtype=torch.float32)

########################################################################################################################
# 2. Define Dynamics Model, Reward Function, and Initial Distributions
########################################################################################################################

def compute_fx(x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Computes the derivatives fx given state x and action.
    
    Parameters
    ----------
    x : torch.Tensor
        Current state x, shape [batch_size, 4].
    action : torch.Tensor
        Current action, shape [batch_size, 2].
    
    Returns
    -------
    fx : torch.Tensor
        Derivatives, shape [batch_size, 4].
    """
    u0 = action[:, 0]
    u1 = action[:, 1]
    kmod = mod_min + 0.5 * (u0 + 1.0) * (mod_max - mod_min)
    kphase = phase_min + 0.5 * (u1 + 1.0) * (phase_max - phase_min)
    
    x0, x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    
    fx0 = (ire * (lmbda_re * x0 - lmbda_cx * x1)
           - (mu_re * x0 - mu_cx * x1) * (x0 ** 2 + x1 ** 2)
           + (alpha_re * x2 - alpha_cx * x3)
           + x0 * kmod * torch.cos(kphase)
           - x1 * kmod * torch.sin(kphase))
    
    fx1 = (ire * (lmbda_re * x1 + lmbda_cx * x0)
           - (mu_re * x1 + mu_cx * x0) * (x0 ** 2 + x1 ** 2)
           + (alpha_re * x3 + alpha_cx * x2)
           + x0 * kmod * torch.sin(kphase)
           + x1 * kmod * torch.cos(kphase))
    
    fx2 = (-omega_f * gamma * x2
           - domega * x3
           + beta_m * x0)
    
    fx3 = (-omega_f * gamma * x3
           + domega * x2
           + beta_m * x1)
    
    fx = torch.stack([fx0, fx1, fx2, fx3], dim=1)
    
    return fx

def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Computes the next state given the current state and action using Low-Storage Runge-Kutta 4.
    
    Parameters
    ----------
    state : torch.Tensor
        Current state tensor of shape [batch_size, 8] (x and fx).
    action : torch.Tensor
        Action tensor of shape [batch_size, 2].
    
    Returns
    -------
    next_state : torch.Tensor
        Next state tensor of shape [batch_size, 8].
    """
    x = state[:, :4]      # [batch_size, 4]
    fx = state[:, 4:]     # [batch_size, 4]
    
    # Initialize Runge-Kutta variables
    u = torch.zeros_like(x)  # [batch_size, 4]
    k = torch.zeros_like(x)  # [batch_size, 4]
    
    # Iterate through LSRK4 stages
    for j in range(5):
        # Compute intermediate state
        x_stage = x + lsrk4_a[j] * u  # [batch_size, 4]
        
        # Compute derivatives at intermediate state
        fx_stage = compute_fx(x_stage, action)  # [batch_size, 4]
        
        # Update u and k
        u = lsrk4_a[j] * u + dt_act * fx_stage  # [batch_size, 4]
        k = k + lsrk4_b[j] * u  # [batch_size, 4]
    
    # Update state
    x_new = x + k  # [batch_size, 4]
    fx_new = compute_fx(x_new, action)  # [batch_size, 4]
    
    # Form next state by concatenating x_new and fx_new
    next_state = torch.cat([x_new, fx_new], dim=1)  # [batch_size, 8]
    
    return next_state

def rewards(state: torch.Tensor, action: torch.Tensor, step_count: int) -> torch.Tensor:
    """
    Computes the reward given the current state and action.
    
    Parameters
    ----------
    state : torch.Tensor
        Current state tensor of shape [batch_size, 8].
    action : torch.Tensor
        Action tensor of shape [batch_size, 2].
    step_count : int
        Current time step count.
    
    Returns
    -------
    reward : torch.Tensor
        Reward tensor of shape [batch_size,].
    """
    x = state[:, :4]  # [batch_size, 4]
    # fx = state[:, 4:]
    
    t = step_count * dt_act  # Scalar
    
    # Compute y_prev
    x2 = x[:, 2]
    x3 = x[:, 3]
    y_prev = 2.0 * (x2 * torch.cos(omega_f * t) - x3 * torch.sin(omega_f * t))  # [batch_size]
    
    # Compute next state
    next_state = dynamics(state, action, step_count)  # [batch_size, 8]
    x_new = next_state[:, :4]  # [batch_size, 4]
    
    # Compute y_curr
    t_new = (step_count + 1) * dt_act  # Scalar
    x2_new = x_new[:, 2]
    x3_new = x_new[:, 3]
    y_curr = 2.0 * (x2_new * torch.cos(omega_f * t_new) - x3_new * torch.sin(omega_f * t_new))  # [batch_size]
    
    # Compute cost associated with the action
    x0_new = x_new[:, 0]
    x1_new = x_new[:, 1]
    u0 = action[:, 0]
    u1 = action[:, 1]
    kmod = mod_min + 0.5 * (u0 + 1.0) * (mod_max - mod_min)  # [batch_size]
    kphase = phase_min + 0.5 * (u1 + 1.0) * (phase_max - phase_min)  # [batch_size]
    
    term1 = x0_new * torch.cos(omega_f * t_new) - x1_new * torch.sin(omega_f * t_new)  # [batch_size]
    term2 = x1_new * torch.cos(omega_f * t_new) + x0_new * torch.sin(omega_f * t_new)  # [batch_size]
    
    cost = (2.0 * kmod * torch.cos(kphase) * term1 - 
            2.0 * kmod * torch.sin(kphase) * term2)  # [batch_size]
    cost = 0.5 * cost ** 2  # [batch_size]
    
    # Compute the reward
    reward = (2.0 * omega_s * gamma * ((y_curr - y_prev) / dt_act) ** 2
              - weight * cost)  # [batch_size]
    
    return reward

def initial_distribution(batch_size: int) -> torch.Tensor:
    """
    Generates the initial state distribution.
    
    Parameters
    ----------
    batch_size : int
        Number of samples to generate.
    
    Returns
    -------
    initial_state : torch.Tensor
        Initial state tensor of shape [batch_size, 8].
    """
    # Initialize x with fixed initial values
    x0 = -0.00385 * torch.ones(batch_size)
    x1 = -0.00378 * torch.ones(batch_size)
    x2 = 0.00118 * torch.ones(batch_size)
    x3 = -0.00131 * torch.ones(batch_size)
    x = torch.stack([x0, x1, x2, x3], dim=1)  # [batch_size, 4]
    
    # Compute fx for initial state with zero action
    action = torch.zeros(batch_size, action_dim)  # [batch_size, 2]
    fx = compute_fx(x, action)  # [batch_size, 4]
    
    # Form the initial state by concatenating x and fx
    initial_state = torch.cat([x, fx], dim=1)  # [batch_size, 8]
    
    return initial_state

def rand_distribution(batch_size: int) -> torch.Tensor:
    """
    Generates a random state distribution within the specified state range.
    
    Parameters
    ----------
    batch_size : int
        Number of samples to generate.
    
    Returns
    -------
    rand_state : torch.Tensor
        Random state tensor of shape [batch_size, 8].
    """
    # Randomly initialize x within [-0.05, 0.05] for each dimension
    x = (0.05 - (-0.05)) * torch.rand(batch_size, 4) + (-0.05)  # [batch_size, 4]
    
    # Compute fx for random x with zero action
    action = torch.zeros(batch_size, action_dim)  # [batch_size, 2]
    fx = compute_fx(x, action)  # [batch_size, 4]
    
    # Form the random state by concatenating x and fx
    rand_state = torch.cat([x, fx], dim=1)  # [batch_size, 8]
    
    return rand_state
