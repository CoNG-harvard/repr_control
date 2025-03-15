import torch
import numpy as np

########################################################################################################################
# 1. define problem-related constants
########################################################################################################################
state_dim = 500                    # State dimension (number of spatial points)
action_dim = 1                     # Action dimension
state_range = [[-np.inf] * state_dim, [np.inf] * state_dim]  # State bounds (unbounded for simplicity)
action_range = [[-1.0], [1.0]]     # Action bounds (between -1 and 1)
max_step = 200                     # Maximum rollout steps per episode
sigma = 0.1                        # Noise standard deviation
env_name = 'Burgers'
assert len(action_range[0]) == len(action_range[1]) == action_dim

########################################################################################################################
# 2. define dynamics model, reward function and initial distribution.
########################################################################################################################
def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Dynamics function for the Burgers' equation environment.

    Parameters
    ----------
    state : torch.Tensor, [batch_size, state_dim]
        The current state of the system (u at all spatial points).
    action : torch.Tensor, [batch_size, action_dim]
        The action applied to the system.

    Returns
    -------
    next_state : torch.Tensor, [batch_size, state_dim]
        The next state of the system after applying the action.
    """
    batch_size, nx = state.shape
    L = 2.0
    dx = L / nx
    dt = 0.2 * dx
    dt_act = 0.05
    ndt_act = int(dt_act / dt)
    amp = 10.0
    u_target = 0.5
    ctrl_xpos = 1.0
    ctrl_pos = int(ctrl_xpos / dx)

    # Initialize fields
    u = state.clone()
    up = u.clone()
    upp = u.clone()
    du = torch.zeros_like(u)
    rhs_field = torch.zeros_like(u)

    # Create noise
    noise = torch.FloatTensor(batch_size, 1).uniform_(-sigma, sigma)  # Shape [batch_size, 1]

    # Scale action
    scaled_action = action * amp  # Shape [batch_size, 1]

    for _ in range(ndt_act):

        # Update previous fields
        upp = up.clone()
        up = u.clone()

        # Boundary conditions
        u[:, 0] = u_target + noise.squeeze(1)
        u[:, -1] = u[:, -2]

        # Compute spatial derivative
        derx(u, du, nx, dx)

        # Build rhs
        rhs(u, du, rhs_field, nx)

        # Add control
        rhs_field[torch.arange(batch_size), ctrl_pos] += scaled_action.squeeze(1)

        # Update u
        dert(u, up, upp, rhs_field, nx, dt)

    next_state = u

    return next_state

def rewards(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Reward function for the Burgers' equation environment.

    Parameters
    ----------
    state : torch.Tensor, [batch_size, state_dim]
    action : torch.Tensor, [batch_size, action_dim]

    Returns
    -------
    rewards : torch.Tensor, [batch_size,]
    """
    batch_size, nx = state.shape
    L = 2.0
    dx = L / nx
    u_target = 0.5
    ctrl_xpos = 1.0
    ctrl_pos = int(ctrl_xpos / dx)

    # Compute the reward
    diff = torch.abs(state[:, ctrl_pos:] - u_target)
    rwd = -torch.sum(diff, dim=1) * dx

    return rwd

def initial_distribution(batch_size: int) -> torch.Tensor:
    """
    Generate initial states for the Burgers' equation environment.

    Parameters
    ----------
    batch_size : int

    Returns
    -------
    state : torch.Tensor, [batch_size, state_dim]
    """
    nx = state_dim
    u_target = 0.5
    state = torch.ones(batch_size, nx) * u_target
    return state

def rand_distribution(batch_size: int) -> torch.Tensor:
    """
    Generate random states for the Burgers' equation environment.

    Parameters
    ----------
    batch_size : int

    Returns
    -------
    state : torch.Tensor, [batch_size, state_dim]
    """
    nx = state_dim
    # For simplicity, we assume the state can vary between -1.0 and 1.0
    state = torch.FloatTensor(batch_size, nx).uniform_(-1.0, 1.0)
    return state

########################################################################################################################
# 3. define helper functions for the dynamics
########################################################################################################################
def derx(u, du, nx, dx):
    """
    Compute the first derivative using a TVD scheme.

    Parameters
    ----------
    u : torch.Tensor, [batch_size, nx]
        Current state.
    du : torch.Tensor, [batch_size, nx]
        Derivative to compute.
    nx : int
        Number of spatial points.
    dx : float
        Spatial step size.
    """
    batch_size = u.shape[0]
    fp  = torch.zeros_like(u)
    fm  = torch.zeros_like(u)
    phi = torch.zeros_like(u)
    r   = torch.zeros_like(u)

    numerator   = u[:, 1:nx-1] - u[:, 0:nx-2]
    denominator = u[:, 2:nx] - u[:, 1:nx-1] + 1.0e-8
    r_value = numerator / denominator
    r[:, 1:nx-1] = r_value

    phi_value = (r_value + torch.abs(r_value)) / (1.0 + r_value)
    phi[:, 1:nx-1] = phi_value

    fp[:, 1:nx-1] = u[:, 1:nx-1] + 0.5 * phi_value * (u[:, 2:nx] - u[:, 1:nx-1])  # f_m+1/2
    fm[:, 1:nx-1] = u[:, 0:nx-2] + 0.5 * phi[:, 0:nx-2] * (u[:, 1:nx-1] - u[:, 0:nx-2])  # f_m-1/2
    du[:, 1:nx-1] = (fp[:, 1:nx-1] - fm[:, 1:nx-1]) / dx

def dert(u, up, upp, rhs_field, nx, dt):
    """
    Compute the time derivative.

    Parameters
    ----------
    u : torch.Tensor, [batch_size, nx]
        Current state to update.
    up : torch.Tensor, [batch_size, nx]
        State at previous time step.
    upp : torch.Tensor, [batch_size, nx]
        State at two time steps before.
    rhs_field : torch.Tensor, [batch_size, nx]
        Right-hand side field.
    nx : int
        Number of spatial points.
    dt : float
        Time step size.
    """
    u[:, 1:nx-1] = (4.0 * up[:, 1:nx-1] - upp[:, 1:nx-1] - 2.0 * dt * rhs_field[:, 1:nx-1]) / 3.0

def rhs(u, du, r, nx):
    """
    Compute the right-hand side of the equation.

    Parameters
    ----------
    u : torch.Tensor, [batch_size, nx]
        Current state.
    du : torch.Tensor, [batch_size, nx]
        Spatial derivative.
    r : torch.Tensor, [batch_size, nx]
        Right-hand side to compute.
    nx : int
        Number of spatial points.
    """
    # Avoid inplace operation on r
    r_new = r.clone()
    r_new[:, 1:nx-1] = u[:, 1:nx-1] * du[:, 1:nx-1]


    return r_new
