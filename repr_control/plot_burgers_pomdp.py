import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob

# Import necessary components from your project
from define_problem_burgers import dynamics, rewards, initial_distribution
from repr_control.agent.randomized_sac import svd_sac_agent_pomdp
from repr_control.envs.custom_env import CustomVecEnv  # Adjust the import path as necessary
import gymnasium

# ########################################################################################################################
# # 1. Define Problem-Related Constants
# ########################################################################################################################

# Define the same constants as in your training code
state_dim = 500                    # Number of spatial points
L = 2.0                            # Domain length
dx = L / state_dim                 # Spatial step size
ctrl_xpos = 1.0                    # Position of control point
ctrl_pos = int(ctrl_xpos / dx)     # Index of control position
burgers_sigma = 0.1                # Sigma for Burgers' equation noise
action_dim = 1                     # Action dimension
state_range = [[0] * state_dim, [2] * state_dim]  # State bounds
action_range = [[-1.0], [1.0]]     # Action bounds
sigma = 0.0                        # Additional noise parameter (if any)
obs_dim = 5                        # Observation dimension (as per your agent's configuration)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1")

# ########################################################################################################################
# # 2. Define Helper Functions
# ########################################################################################################################

def derx(u: torch.Tensor, nx: int, dx: float) -> torch.Tensor:
    """
    Compute the first derivative using a TVD scheme.
    
    Parameters
    ----------
    u : torch.Tensor, [batch_size, nx]
        Current state.
    nx : int
        Number of spatial points.
    dx : float
        Spatial step size.
    
    Returns
    -------
    du : torch.Tensor, [batch_size, nx]
        Spatial derivative of u.
    """
    du = torch.zeros_like(u)

    # Compute r = (u_m - u_{m-1}) / (u_{m+1} - u_m + epsilon)
    epsilon = 1.0e-8
    numerator = u[:, 1:nx-1] - u[:, 0:nx-2]
    denominator = u[:, 2:nx] - u[:, 1:nx-1] + epsilon
    r = numerator / denominator

    # Compute phi = (r + |r|) / (1 + r)
    phi = (r + torch.abs(r)) / (1.0 + r)

    # Compute fluxes fp and fm
    fp = torch.zeros_like(u)
    fm = torch.zeros_like(u)
    fp[:, 1:nx-1] = u[:, 1:nx-1] + 0.5 * phi * (u[:, 2:nx] - u[:, 1:nx-1])
    fm[:, 1:nx-1] = u[:, 0:nx-2] + 0.5 * phi[:, 0:nx-2] * (u[:, 1:nx-1] - u[:, 0:nx-2])

    # Compute spatial derivative du = (fp - fm) / dx
    du[:, 1:nx-1] = (fp[:, 1:nx-1] - fm[:, 1:nx-1]) / dx

    return du

def rhs(u: torch.Tensor, du: torch.Tensor, nx: int) -> torch.Tensor:
    """
    Compute the right-hand side of the inviscid Burgers' equation: rhs = u * du
    
    Parameters
    ----------
    u : torch.Tensor, [batch_size, nx]
        Current state.
    du : torch.Tensor, [batch_size, nx]
        Spatial derivative of u.
    nx : int
        Number of spatial points.
    
    Returns
    -------
    rhs_field : torch.Tensor, [batch_size, nx]
        Right-hand side field.
    """
    rhs_field = torch.zeros_like(u)
    rhs_field[:, 1:nx-1] = u[:, 1:nx-1] * du[:, 1:nx-1]
    return rhs_field

def dert(u: torch.Tensor, up: torch.Tensor, upp: torch.Tensor, rhs_field: torch.Tensor, nx: int, dt: float) -> torch.Tensor:
    """
    Compute the time derivative using a multi-step method:
    u^{n+1} = (4 * u^n - u^{n-1} - 2 * dt * rhs) / 3
    
    Parameters
    ----------
    u : torch.Tensor, [batch_size, nx]
        Current state to update.
    up : torch.Tensor, [batch_size, nx]
        State at previous time step (u^n).
    upp : torch.Tensor, [batch_size, nx]
        State at two time steps before (u^{n-1}).
    rhs_field : torch.Tensor, [batch_size, nx]
        Right-hand side field.
    nx : int
        Number of spatial points.
    dt : float
        Time step size.
    
    Returns
    -------
    u_new : torch.Tensor, [batch_size, nx]
        Updated state (u^{n+1}).
    """
    u_new = (4.0 * up - upp - 2.0 * dt * rhs_field) / 3.0
    return u_new

def simulate_trajectory(agent, env, num_steps=200):
    """
    Simulate a single trajectory using the provided agent and environment.
    
    Parameters:
    - agent: The trained agent with a loaded actor.
    - env: The environment to simulate in.
    - num_steps: Number of steps to simulate.
    
    Returns:
    - trajectory: List of state tensors.
    """
    state, _ = env.reset()
    trajectory = [state.clone()]
    
    for _ in range(num_steps):
        # Select action using the agent's policy
        obs = state[:,ctrl_pos - obs_dim:ctrl_pos]
        action = agent.batch_select_action(obs, explore=False)  # Set explore=False for deterministic actions
        
        # Perform action in the environment
        next_state, reward, terminated, truncated, rollout_info = env.step(action)
        
        trajectory.append(next_state.clone())
        
        state = next_state.clone()
        
        if truncated or terminated:
            break
    
    return trajectory

# ########################################################################################################################
# # 3. Plotting Function
# ########################################################################################################################

def plot_trajectory(trajectory, x, ctrl_xpos, step, output_folder):
    """
    Plot and save the trajectory of u(t, x).
    
    Parameters:
    - trajectory: List of state tensors.
    - x: Spatial grid.
    - ctrl_xpos: Position of the control point.
    - step: Training step corresponding to this trajectory.
    - output_folder: Directory to save the plot.
    """
    num_steps = len(trajectory)
    trajectory_np = torch.stack(trajectory).squeeze(1).cpu().numpy()  # Shape: [num_steps, state_dim]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(x, trajectory_np[0], color='blue', label='u(t,x)')
    ax.set_xlim(0, L)
    ax.set_ylim(min(state_range[0]) - 0.1, max(state_range[1]) + 0.1)
    ax.set_xlabel('Spatial Domain (x)')
    ax.set_ylabel('Solution (u)')
    ax.set_title(f'Trajectory of u(t, x) - Step {step}')
    
    # Add a line indicating the control position
    ax.axvline(x=ctrl_xpos, color='red', linestyle='--', label='Control Position')
    ax.legend()
    
    # Function to update the plot for each frame (optional: create animation)
    def update(frame):
        line.set_ydata(trajectory_np[frame])
        ax.set_title(f'Trajectory of u(t, x) - Step {step}, Frame {frame}/{num_steps}')
        return line,
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=num_steps, blit=True, interval=50, repeat=False)
    
    # Save the animation as a GIF
    gif_filename = os.path.join(output_folder, f'burgers_trajectory_step_{step}.gif')
    try:
        ani.save(gif_filename, writer='imagemagick')
        print(f"Animation saved as {gif_filename}")
    except Exception as e:
        print(f"Failed to save animation as GIF. Error: {e}")
        print("Attempting to save as MP4 instead.")
        mp4_filename = os.path.join(output_folder, f'burgers_trajectory_step_{step}.mp4')
        try:
            ani.save(mp4_filename, writer='ffmpeg')
            print(f"Animation saved as {mp4_filename}")
        except Exception as e2:
            print(f"Failed to save animation as MP4. Error: {e2}")
    
    plt.close(fig)  # Close the figure to free memory

# ########################################################################################################################
# # 4. Main Plotting Logic
# ########################################################################################################################

def main():
    """
    Main function to load the best actor, simulate a trajectory, and plot it.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Plot trajectory using the best actor from training logs.")
    parser.add_argument("--log_path", type=str, required=True,
                        help="Path to the training log directory containing 'best_actor.pth'.")
    parser.add_argument("--plot_folder", type=str, default="burgers_plots",
                        help="Folder to save the trajectory plots.")
    parser.add_argument("--num_steps", type=int, default=200,
                        help="Number of steps to simulate for the trajectory.")
    args = parser.parse_args()

    log_path = args.log_path
    plot_folder = args.plot_folder
    num_steps = args.num_steps

    # Create the 'burgers_plots' folder if it doesn't exist
    os.makedirs(plot_folder, exist_ok=True)

    # Path to the best actor's checkpoint
    best_actor_path = os.path.join(log_path, "best_actor.pth")
    best_critic_last_layer_path = os.path.join(log_path, "best_critic_last_layer.pth")
    best_critic_phi_path = os.path.join(log_path, "best_critic_phi.pth")

    # Check if the best actor checkpoint exists
    if not os.path.exists(best_actor_path):
        print(f"Best actor checkpoint not found at {best_actor_path}.")
        return

    # Initialize the agent with the same parameters as during training
    # Ensure that you replicate the initialization parameters
    # Here, kwargs should be loaded or set to match your training configuration
    # For demonstration, we'll assume default values; modify as necessary

    # Load training parameters
    train_params_path = os.path.join(log_path, "train_params.pth")
    if not os.path.exists(train_params_path):
        print(f"Training parameters not found at {train_params_path}.")
        return

    kwargs = torch.load(train_params_path, map_location=device)

    # Initialize the agent
    agent = svd_sac_agent_pomdp.svdSACAgent(
        dynamics_fn=dynamics,
        rewards_fn=rewards,
        critic_phi=None,  # Assuming critic_phi was saved separately
        use_V_critic=kwargs.get("use_V_critic", False),
        sigma=kwargs.get("sigma", 0.0),
        obs_dim=kwargs.get("obs_dim", 5),
        # rsvd_num = 512,
        **kwargs
    )

    # Load the best actor's state dictionary
    agent.actor.load_state_dict(torch.load(best_actor_path, map_location=device))
    agent.actor.eval()  # Set to evaluation mode

    # Load the critic's last layer and phi_net if necessary
    if os.path.exists(best_critic_last_layer_path):
        agent.critic_last_layer.load_state_dict(torch.load(best_critic_last_layer_path, map_location=device))
    if os.path.exists(best_critic_phi_path):
        agent.phi_net.load_state_dict(torch.load(best_critic_phi_path, map_location=device))

    # Initialize the environment
    env = CustomVecEnv(
        dynamics=dynamics,
        rewards=rewards,
        initial_distribution=initial_distribution,
        rand_distribution=initial_distribution,  # Adjust if you have a separate rand_distribution
        state_range=state_range,
        action_range=action_range,
        sigma=sigma,
        sample_batch_size=1,  # Single simulation
        device=device,
        max_episode_steps=200,  # Adjust as needed
    )

    # Simulate a trajectory
    trajectory = simulate_trajectory(agent, env, num_steps=num_steps)

    # Create the correct spatial grid
    x = np.arange(state_dim) * dx  # Correct spatial grid from 0 to L-dx

    # Plot and save the trajectory
    # Assuming that <exp_name> is unique per training run
    # You can include the log_path's identifier in the plot title or filename if desired
    step_identifier = "best_actor"  # Since we're using the best actor overall
    plot_trajectory(trajectory, x, ctrl_xpos, step_identifier, plot_folder)

    print("Trajectory plotting completed.")

# ########################################################################################################################
# # 5. Execute Plotting
# ########################################################################################################################

if __name__ == "__main__":
    main()
