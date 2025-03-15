import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse

# Import your environment and dynamics functions
from define_problem_burgers import dynamics, initial_distribution, state_dim, action_dim, dx, L, ctrl_xpos, action_range, state_range, rewards
from repr_control.envs.custom_env import CustomVecEnv  # Adjust the import path as necessary
device = torch.device("cpu")

def simulate_trajectory(agent, env, device, num_steps=500):
    """
    Simulate a single trajectory using the provided agent and environment.

    Parameters
    ----------
    agent : YourAgentClass
        The agent with a method `batch_select_action`.
    env : CustomVecEnv
        The environment to interact with.
    device : torch.device
        The device to perform computations on.
    num_steps : int
        Number of steps to simulate.

    Returns
    -------
    trajectory : list of np.ndarray
        List containing the state at each time step.
    """
    state, _ = env.reset()
    trajectory = [state.clone().squeeze(0).cpu().numpy()]  # Initial state

    for _ in range(num_steps):
        # Select action without exploration
        action = agent.batch_select_action(state, explore=False)  # Shape: [1, 1]
        
        # Perform action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Store next state
        trajectory.append(next_state.clone().squeeze(0).cpu().numpy())
        
        # Update state
        state = next_state.clone()
        
        # # Check for termination
        # if terminated.any() or truncated.any():
        #     break

    return trajectory

def plot_and_save_trajectory(trajectory, x, output_folder):
    """
    Plot the trajectory and save it as an animation.

    Parameters
    ----------
    trajectory : list of np.ndarray
        List containing the state at each time step.
    x : np.ndarray
        Spatial grid points.
    step : int
        The step number corresponding to the best actor.
    output_folder : str
        Path to the folder where plots will be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(x, trajectory[0], color='blue', label='u(t,x)')
    ax.set_xlim(0, L)
    ax.set_ylim(state_range[0][0] - 0.1, state_range[1][0] + 0.1)
    ax.set_xlabel('Spatial Domain (x)')
    ax.set_ylabel('Solution (u)')
    ax.set_title(f'Trajectory of u(t, x)')
    ax.axvline(x=ctrl_xpos, color='red', linestyle='--', label='Control Position')
    ax.legend()

    def update(frame):
        line.set_ydata(trajectory[frame])
        ax.set_title(f'Trajectory of u(t, x), Time {frame}')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), blit=True, interval=50, repeat=False)

    # Save the animation as a GIF
    gif_filename = os.path.join(output_folder, f'burgers_trajectory.gif')
    try:
        ani.save(gif_filename, writer='imagemagick')
        print(f"Saved animation as {gif_filename}")
    except Exception as e:
        print(f"Failed to save GIF: {e}")
        print("Attempting to save as MP4 instead.")
        mp4_filename = os.path.join(output_folder, f'burgers_trajectory_step.mp4')
        try:
            ani.save(mp4_filename, writer='ffmpeg')
            print(f"Saved animation as {mp4_filename}")
        except Exception as e2:
            print(f"Failed to save MP4: {e2}")

    plt.close(fig)

def main(log_path, output_folder='burgers_plots', num_steps=500):
    """
    Main function to plot trajectories using best actors saved at evaluation steps.

    Parameters
    ----------
    log_path : str
        Path to the log directory where best_actor_step_{t}.pth files are saved.
    output_folder : str
        Path to the folder where plots will be saved.
    num_steps : int
        Number of steps to simulate for each trajectory.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all best_actor_step_{t}.pth files
    best_actor_files = [f for f in os.listdir(log_path) if f.startswith('best_actor') and f.endswith('.pth')]
    # best_actor_files.sort(key=lambda x: int(x.split('_')[-1].split('.pth')[0]))  # Sort by step number

    if not best_actor_files:
        print(f"No best_actor_step_{{t}}.pth files found in {log_path}")
        return

    # Create spatial grid
    x = np.arange(state_dim) * dx  # Corrected spatial grid

    for actor_file in best_actor_files: #should just be 1 file
        # Extract step number
        # step = int(actor_file.split('_')[-1].split('.pth')[0])

        # Load best actor
        actor_path = os.path.join(log_path, actor_file)
        print(f"Loading actor from {actor_path}")
        best_actor_state_dict = torch.load(actor_path, map_location=device)

        # Initialize agent and load state_dict
        # This assumes that your agent has an 'actor' attribute and a method to load the state_dict
        # Adjust according to your actual agent implementation
        # Example:
        # agent = sac_agent.SACAgent(...)  # Initialize with necessary parameters
        # agent.actor.load_state_dict(best_actor_state_dict)

        # For demonstration, we'll assume the agent has already been initialized and is ready to load the actor
        # Replace the following lines with your actual agent initialization and loading logic
        # from repr_control.agent.rfsac import rfsac_agent  # Adjust import as necessary
        from repr_control.agent.randomized_sac import random_sac_agent, svd_sac_agent_pomdp

        # Initialize agent with the same parameters used during training
        # You might need to load the training parameters or define them here
        # For simplicity, let's assume default parameters; adjust as necessary
                agent = svd_sac_agent_pomdp.svdSACAgent(dynamics_fn = dynamics, rewards_fn = rewards, critic_phi = None, 
        use_V_critic = use_V_critic, sigma = sigma,obs_dim = obs_dim,**kwargs)
        agent = svd_sac_agent_pomdp.svdSACAgent(
            dynamics_fn=dynamics,
            rewards_fn=rewards,  # Not needed for plotting
            state_dim=state_dim,
            action_dim=action_dim,
            action_range=action_range,
            device=device,
            rf_num=512,  # Adjust if different
            nystrom_sample_dim=8192,  # Adjust if different
            hidden_dim=256,  # Adjust if different
            feature_dim=256,  # Adjust if different
            discount=0.99,
            tau=0.005,
            embedding_dim=-1,
            use_nystrom=False  # Adjust based on your training
        )

        # Load the actor state_dict
        agent.actor.load_state_dict(best_actor_state_dict)
        agent.actor.to(device)
        agent.actor.eval()  # Set to evaluation mode

        # Initialize environment for simulation
        from repr_control.envs.custom_env import CustomVecEnv  # Adjust import as necessary

        env = CustomVecEnv(
            dynamics=dynamics,
            rewards=rewards,  # If used in your CustomVecEnv
            initial_distribution=initial_distribution,
            state_range=state_range,
            action_range=action_range,
            sigma=0.0,  # Adjust if needed
            sample_batch_size=1,  # Single simulation
            device=device,
        )

        # Simulate trajectory
        trajectory = simulate_trajectory(agent, env, device, num_steps=num_steps)

        # Plot and save trajectory
        plot_and_save_trajectory(trajectory, x,output_folder)

    print("All trajectories have been plotted and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot trajectories from trained best actors.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to the log directory containing best_actor.pth files.")
    parser.add_argument("--output_folder", type=str, default="burgers_plots_trained", help="Folder to save the trajectory plots.")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of steps to simulate for each trajectory.")

    args = parser.parse_args()

    main(log_path=args.log_path, output_folder=args.output_folder, num_steps=args.num_steps)
