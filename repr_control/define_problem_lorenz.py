import torch
import numpy as np

########################################################################################################################
# 1. Define problem-related constants
########################################################################################################################
state_dim = 3                      # State dimension for the Lorenz system (x, y, z)
action_dim = 1                     # Action dimension (now continuous in [-1, 1])
state_range = [[-np.inf] * state_dim, [np.inf] * state_dim]  # State bounds (unbounded for simplicity)
action_range = [[-1.0], [1.0]]     # Action bounds
max_step = 500                     # Maximum rollout steps per episode
sigma_l = 10.0                       # Lorenz parameter sigma
rho = 28.0                         # Lorenz parameter rho
beta = 8.0 / 3.0                   # Lorenz parameter beta
dt = 0.05                          # Time step
dt_act = 0.05                      # Action time step
ndt_act = int(dt_act / dt)         # Number of numerical steps per action
sigma = 0.1                         #noise sigma
env_name = 'Lorenz'
assert len(action_range[0]) == len(action_range[1]) == action_dim

########################################################################################################################
# 2. Define dynamics model, reward function, and initial distribution
########################################################################################################################
def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Dynamics function for the Lorenz system with continuous actions in [-1, 1].

    Parameters
    ----------
    state : torch.Tensor, [batch_size, state_dim]
        The current state of the system (x, y, z).
    action : torch.Tensor, [batch_size, action_dim]
        The action applied to the system (continuous values in [-1, 1]).

    Returns
    -------
    next_state : torch.Tensor, [batch_size, state_dim]
        The next state of the system after applying the action.
    """
    batch_size, _ = state.shape
    device = state.device
    dtype = state.dtype

    # Ensure action is within [-1, 1]
    action = torch.clamp(action, -1.0, 1.0)

    # Actions are continuous values in [-1, 1]
    u = action.squeeze()  # Shape: [batch_size]

    # Define the integrator coefficients (lsrk4)
    a = torch.tensor([0.0, -0.417890474499852,
                      -1.192151694642677, -1.697784692471528,
                      -1.514183444257156], dtype=dtype, device=device)
    b = torch.tensor([0.149659021999229, 0.379210312999627,
                      0.822955029386982, 0.699450455949122,
                      0.153057247968152], dtype=dtype, device=device)

    steps = len(a)

    x = state.clone()    # Current state
    xk = x.clone()       # Accumulated state (equivalent to uk in original code)
    u_storage = torch.zeros_like(x)  # Storage for the u variable in the integrator

    # Run solver over ndt_act time steps
    for _ in range(ndt_act):

        # Reset temporary variables for the integration
        x_temp = x.clone()
        xk = x.clone()  # Reset xk for each ndt_act

        # Loop over Runge-Kutta stages
        for j in range(steps):

            # Compute rhs without forcing term
            fx = torch.zeros_like(x_temp)
            fx[:, 0] = sigma_l * (x_temp[:, 1] - x_temp[:, 0])
            fx[:, 1] = x_temp[:, 0] * (rho - x_temp[:, 2]) - x_temp[:, 1]
            fx[:, 2] = x_temp[:, 0] * x_temp[:, 1] - beta * x_temp[:, 2]

            # Add forcing term to fx[:, 1]
            fx[:, 1] += u

            # Update u_storage (equivalent to 'u' in original code)
            u_storage = a[j] * u_storage + dt * fx

            # Update xk
            xk = xk + b[j] * u_storage

            # Prepare for next stage
            x_temp = xk.clone()

        # Update the state after ndt_act steps
        x = xk.clone()

    next_state = x
    return next_state

def rewards(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Reward function for the Lorenz system.

    Parameters
    ----------
    state : torch.Tensor, [batch_size, state_dim]
    action : torch.Tensor, [batch_size, action_dim]

    Returns
    -------
    rewards : torch.Tensor, [batch_size,]
    """
    # Reward is 1.0 if x[0] < 0, else 0.0
    rewards = torch.where(state[:, 0] < 0.0, 1.0, 0.0)
    return rewards

def initial_distribution(batch_size: int) -> torch.Tensor:
    """
    Generate initial states for the Lorenz system.

    Parameters
    ----------
    batch_size : int

    Returns
    -------
    state : torch.Tensor, [batch_size, state_dim]
    """
    state = torch.full((batch_size, state_dim), 10.0)
    return state

def rand_distribution(batch_size: int) -> torch.Tensor:
    """
    Generate random states for the Lorenz system.

    Parameters
    ----------
    batch_size : int

    Returns
    -------
    state : torch.Tensor, [batch_size, state_dim]
    """
    # For the Lorenz system, we can initialize states randomly within a reasonable range
    state = torch.FloatTensor(batch_size, state_dim).uniform_(-20.0, 20.0)
    return state
