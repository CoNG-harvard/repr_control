import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

########################################################################################################################
# 1. define problem-related constants
########################################################################################################################
state_dim = 500                    # State dimension (number of spatial points)
obs_dim = 6 #make it even
action_dim = 1                     # Action dimension
state_range = [[0] * state_dim, [2] * state_dim]  # State bounds (unbounded for simplicity)
action_range = [[-1.0], [1.0]]     # Action bounds (between -1 and 1)
max_step = 200                      # Maximum rollout steps per episode
burgers_sigma = 0.1                     #sigma for burgers
sigma = 0.0
env_name = 'Burgers'
#
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

nx = state_dim
L = 2.0
dx = L / nx
ctrl_xpos = 1.0
ctrl_pos = int(ctrl_xpos / dx)
print("ctrl pos", ctrl_pos)
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
    # print("ndt_act",ndt_act)
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
    noise = torch.FloatTensor(batch_size, 1).uniform_(-burgers_sigma, burgers_sigma)  # Shape [batch_size, 1]

    # Scale action
    # print("action", action)
    scaled_action = action * amp  # Shape [batch_size, 1]
    # print("scaled action", scaled_action)

    for _ in range(ndt_act):

        # Update previous fields
        upp = up.clone()
        up = u.clone()

        # Boundary conditions
        # print("noise squeeze 1 shape", noise.squeeze(1).shape)
        u[:, 0] = u_target + noise.squeeze(1)
        # print("u[:0] shape", u[:,0].shape)
        # print("u shape", u.shape)
        u[:, -1] = u[:, -2]

        # Compute spatial derivative
        du = derx(u, du, nx, dx)

        # Build rhs
        rhs_field = rhs(u, du, rhs_field, nx)

        # Add control
        rhs_field[torch.arange(batch_size), ctrl_pos] += scaled_action.squeeze(1)

        # Update u
        u = dert(u, up, upp, rhs_field, nx, dt)

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
    # print("dx",dx)
    u_target = 0.5
    ctrl_xpos = 1.0
    ctrl_pos = int(ctrl_xpos / dx)

    # Compute the reward
    diff = torch.abs(state[:, ctrl_pos:] - u_target)
    # print("diff", diff)
    # print("dynamics state", state)
    # print("diff shape", diff.shape)
    rwd = -torch.sum(diff, dim=1) * dx
    # print("rwd shape", rwd.shape)

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
    print("init state", state)
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

    return du

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

    return u

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


def plot_trajectory():
    """
    Simulate a single run of the inviscid Burgers' equation and plot the trajectory of u(t, x).
    Saves the animation to the 'burgers_plots' folder.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Create the 'burgers_plots' folder if it doesn't exist
    output_folder = 'burgers_plots'
    os.makedirs(output_folder, exist_ok=True)

    # Simulation parameters
    batch_size = 1  # Single run
    num_steps = 200  # Number of rollout steps

    # Initialize state
    state = initial_distribution(batch_size)  # Shape: [1, state_dim]

    # Define actions (e.g., zero actions for simplicity)
    actions = torch.zeros(batch_size, 1, device=device)  # Shape: [1, 1]

    # Store states for plotting
    trajectory = [state.clone()]  # List of tensors

    print("Starting simulation...")
    tot_reward = 0
    opp_const = 10
    for step in range(num_steps):
        # Compute next state
        if state[:,249] > 0.5001:
            actions = torch.ones(batch_size, 1, device=device) * opp_const * (state[:,249] - 0.5)
        elif state[:,249] < 0.4901:
            actions = torch.ones(batch_size, 1, device=device) * -opp_const * (0.5 - state[:,249])
        state = dynamics(state, actions)
        trajectory.append(state.clone())
        if (step + 1) % 50 == 0:
            print(f"Completed {step + 1}/{num_steps} steps")
        tot_reward += rewards(state,actions)
    print("total reward:", tot_reward)

    print("Simulation completed.")

    # Convert trajectory to numpy array for plotting
    # Shape: [num_steps + 1, state_dim]
    trajectory_np = torch.stack(trajectory).squeeze(1).cpu().numpy()

    # Create spatial grid for plotting
    # x = np.linspace(0, L, num=state_dim, endpoint=False) * dx  # [state_dim,]
    x = np.arange(state_dim) * dx


    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(x, trajectory_np[0], color='blue')
    ax.set_xlim(0, L)
    ax.set_ylim(state_range[0][0] - 0.1, state_range[1][0] + 0.1)
    ax.set_xlabel('Spatial Domain (x)')
    ax.set_ylabel('Solution (u)')
    ax.set_title('Trajectory of u(t, x) for Inviscid Burgers\' Equation')

    # Add a line indicating the control position
    ax.axvline(x=ctrl_xpos, color='red', linestyle='--', label='Control Position')
    ax.legend()

    # Function to update the plot for each frame
    def update(frame):
        line.set_ydata(trajectory_np[frame])
        ax.set_title(f'Trajectory of u(t, x) - Step {frame}/{num_steps}')
        return line,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=num_steps + 1, blit=True, interval=50, repeat=False)

    # Save the animation as a GIF
    gif_filename = os.path.join(output_folder, 'burgers_trajectory_opp_const=%d_sigma=%.2f.gif'%(opp_const, burgers_sigma))
    # Ensure ImageMagick is installed for GIF saving, or use a different writer
    try:
        ani.save(gif_filename, writer='imagemagick')
        print(f"Animation saved as {gif_filename}")
    except Exception as e:
        print(f"Failed to save animation as GIF. Error: {e}")
        print("Attempting to save as MP4 instead.")
        mp4_filename = os.path.join(output_folder, 'burgers_trajectory.mp4')
        ani.save(mp4_filename, writer='ffmpeg')
        print(f"Animation saved as {mp4_filename}")

    # Display the animation
    # plt.show()

# ########################################################################################################################
# # 3. Execute Plotting
# ########################################################################################################################

if __name__ == "__main__":
    plot_trajectory()