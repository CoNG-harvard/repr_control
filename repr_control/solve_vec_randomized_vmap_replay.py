import argparse
import os
import pickle as pkl

from tensorboardX import SummaryWriter
from datetime import datetime
from repr_control.utils import util_network, buffer, buffer_vec
from repr_control.agent.sac import sac_agent_network_vmap_v2
from repr_control.agent.randomized_sac import random_sac_agent_network, random_sac_agent_network_phi_s
# from define_problem_kuramoto import *
# from define_problem_kuramoto_v3 import *
# from define_problem_kuramoto_v2_thdot import *
from define_problem_kuramoto_v2_thdot_signed_omega import *
# from define_problem_kuramoto_v2_thdot_2nd_order import *
from gymnasium.envs.registration import register
import gymnasium
import yaml
from repr_control.utils.buffer import Batch
import torch
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### Parameters that might need tuning
    parser.add_argument("--alg", default="randomized_sac",
                        help="The algorithm to use. Options: rfsac, sac, randomized_sac.")
    parser.add_argument("--env", default='custom_vec',
                        help="Name of your environment/dynamics, only for folder names.")
    parser.add_argument("--rf_num", default=512, type=int,
                        help="Number of random features.")
    parser.add_argument("--rsvd_num", default=512, type=int,
                        help="Number of features for randomized SVD.")
    parser.add_argument("--nystrom_sample_dim", default=8192, type=int,
                        help="Sampling dimension for Nystrom critic.")
    parser.add_argument("--device", default='cuda:1', type=str,
                        help="PyTorch device (e.g., 'cuda:0', 'cpu').")

    ### Other parameters
    parser.add_argument("--seed", default=0, type=int,
                        help='Random seed.')
    parser.add_argument("--start_timesteps", default=0, type=int,
                        help='Number of initial steps with random actions.')
    parser.add_argument("--eval_freq", default=100, type=int,
                        help='Frequency of evaluations.')
    parser.add_argument("--max_timesteps", default=1e4, type=int,
                        help='Total number of training steps.')
    parser.add_argument("--batch_size", default=128, type=int,
                        help='Batch size for training.')
    parser.add_argument("--hidden_dim", default=256, type=int,
                        help='Hidden dimension size for networks.')
    parser.add_argument("--feature_dim", default=256, type=int,
                        help='Latent feature dimension.')
    parser.add_argument("--discount", default=0.99, type=float,
                        help='Discount factor.')
    # parser.add_argument("--lr", default=1e-4, type=float,
    #                     help='learning rate.')
    parser.add_argument("--lr", default=2e-4, type=float,
                        help='learning rate.')
    parser.add_argument("--tau", default=0.005, type=float,
                        help='Target network update rate.')
    parser.add_argument("--rsvd_sigma", default=0.05, type=float,
                        help='Sigma for generating random mu in RSVD.')
    parser.add_argument("--embedding_dim", default=-1, type=int,
                        help='Embedding dimension. Set -1 to disable embedding.')
    parser.add_argument("--use_nystrom", action='store_true',
                        help='Use Nystrom method for the critic.')
    parser.add_argument("--use_random_feature", dest='use_nystrom', action='store_false',
                        help='Use random feature method for the critic.')
    parser.set_defaults(use_nystrom=False)


    parser.add_argument("--has_feature_step", action='store_true',
                        help="Enable the feature step in training.")
    parser.add_argument("--no_feature_step", dest='has_feature_step', action='store_false',
                        help="Disable the feature step in training.")
    parser.set_defaults(has_feature_step=True)
    args = parser.parse_args()
    args.device = curr_device

    alg_name = args.alg
    exp_name = f'seed_{args.seed}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    # Setup logging
    use_V_critic = False
    # has_feature_step = True
    log_path = f'log/replay/{alg_name}/{env_name}/N={N}/kappa={eval_kappa}/start_timesteps={args.start_timesteps}/sigma={sigma}/rsvd_sigma={args.rsvd_sigma}/batchsize={args.batch_size}/V_critic={use_V_critic}/feature_step={args.has_feature_step}/{exp_name}'
    summary_writer = SummaryWriter(log_path + "/summary_files")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare arguments
    kwargs = vars(args)
    kwargs.update({
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_range": action_range,
        'obs_space_high': np.clip(state_range[0], -3., 3.).tolist(),
        'obs_space_low': np.clip(state_range[1], -3., 3.).tolist(),
        'device': args.device,
        'rsvd_num': args.rsvd_num,
    })

    #args from define_problem.py
    kwargs.update({
        "N": N,
        'eval_adjacency': eval_adjacency,
        'eval_minus_one_adjacency': eval_minus_one_adjacency,
        'policy_adjacency': policy_adjacency,
        'kappa_obs_dim': kappa_obs_dim,
        "eval_kappa_obs_dim": eval_kappa_obs_dim,
        })

    # Initialize environments
    if args.env == 'custom':
        register(id='custom-v0',
                 entry_point='repr_control.envs:CustomEnv',
                 max_episode_steps=max_step)
        env = gymnasium.make('custom-v0',
                             dynamics=dynamics,
                             rewards=rewards,
                             initial_distribution=initial_distribution,
                             state_range=state_range,
                             action_range=action_range,
                             sigma=sigma)
        eval_env = gymnasium.make('custom-v0',
                                  dynamics=dynamics,
                                  rewards=rewards,
                                  initial_distribution=initial_distribution,
                                  state_range=state_range,
                                  action_range=action_range,
                                  sigma=sigma)
        env = gymnasium.wrappers.RescaleAction(env, min_action=-1, max_action=1)
        eval_env = gymnasium.wrappers.RescaleAction(eval_env, min_action=-1, max_action=1)
    elif args.env == 'custom_vec':
        from repr_control.envs.custom_env import CustomVecEnv
        env = CustomVecEnv(
            dynamics=dynamics,
            rewards=rewards,
            initial_distribution=initial_distribution,
            rand_distribution=None,
            state_range=state_range,
            action_range=action_range,
            sigma=sigma,
            sample_batch_size=args.batch_size,
            device=torch.device(args.device),
        )
        eval_env = CustomVecEnv(
            dynamics=dynamics,
            rewards=rewards,
            initial_distribution=initial_distribution,
            rand_distribution=None,
            state_range=state_range,
            action_range=action_range,
            sigma=sigma,
            sample_batch_size=args.batch_size,
            device=torch.device(args.device),
        )
    else:
        env = gymnasium.make(args.env)
        eval_env = gymnasium.make(args.env)
        env = gymnasium.wrappers.RescaleAction(env, min_action=-1, max_action=1)
        eval_env = gymnasium.wrappers.RescaleAction(eval_env, min_action=-1, max_action=1)

    # Initialize agent
    if args.alg == "randomized_sac":
        agent = random_sac_agent_network.CustomPhiSACAgent(
            dynamics_fn=dynamics,
            rewards_fn=rewards,
            critic_phi = None, 
            use_V_critic = use_V_critic,
            sigma = sigma,
            # eval_adjacency = eval_adjacency,
            # policy_adjacency = policy_adjacency,
            # kappa_obs_dim = kappa_obs_dim,
            # eval_kappa_obs_dim = eval_kappa_obs_dim,
            eval_kappa_action_dim = eval_kappa_action_dim,
            # N = N,
            # has_feature_step = args.has_feature_step,
            **kwargs
        )
        # agent = random_sac_agent_network_phi_s.CustomPhiSACAgent(
        #     dynamics_fn=dynamics,
        #     rewards_fn=rewards,
        #     critic_phi = None, 
        #     use_V_critic = use_V_critic,
        #     sigma = sigma,
        #     eval_adjacency = eval_adjacency,
        #     policy_adjacency = policy_adjacency,
        #     kappa_obs_dim = kappa_obs_dim,
        #     eval_kappa_obs_dim = eval_kappa_obs_dim,
        #     eval_kappa_action_dim = eval_kappa_action_dim,
        #     N = N,
        #     # has_feature_step = args.has_feature_step,
        #     **kwargs
        # )
    elif args.alg == "sac_v2":
        agent = sac_agent_network_vmap_v2.CustomModelSACAgent(
            dynamics_fn=dynamics,
            rewards_fn=rewards,
            **kwargs
        )
    else:
        raise NotImplementedError("Algorithm not implemented.")

    replay_buffer = buffer_vec.ReplayBuffer(state_dim , action_dim, N = N, max_size = int(args.max_timesteps * args.batch_size), device=args.device)


    # Initialize training variables
    evaluations = []
    state, _ = env.reset()
    done = False
    episode_reward = torch.zeros((args.batch_size, N), device=torch.device(args.device))
    episode_timesteps = 0
    episode_num = 0
    timer = util_network.Timer()

    # Best model tracking
    best_eval_reward = -1e6
    best_actors = [None] * N
    best_critic_phi = [None] * N
    best_critic_last_layer = [None] * N

    # Save training parameters
    with open(os.path.join(log_path, 'train_params.yaml'), 'w') as fp:
        yaml.dump(kwargs, fp, default_flow_style=False)

    with open(os.path.join(log_path, 'train_params.pth'), 'wb') as f:
        torch.save(kwargs, f)
        print("PyTorch kwargs saved")

    # Training loop
    for t in range(int(args.max_timesteps + args.start_timesteps)):

        episode_timesteps += 1

        # Select action
        if t < args.start_timesteps:
            action = env.sample_action()
        else:
            action = agent.batch_select_action_network(state, explore=True)


        if t % 50 == 0:
            # print("action at step=%d" %t, action)
            pass

        # Perform action
        next_state, reward, terminated, truncated, rollout_info = env.step(action)
        done = truncated
        replay_buffer.add(state, action, next_state, reward, done)


        # batch = Batch(
        #     state=state,
        #     action=action,
        #     reward=reward,
        #     next_state=next_state,
        #     done=torch.zeros(size=(state.shape[0], 1), device=torch.device(args.device)),
        # )

        state = next_state.clone()
        episode_reward += reward.reshape((-1, N))
        info = {}

        if t >= args.start_timesteps:
            # info = agent.batch_train(batch)
            info = agent.train(replay_buffer,batch_size=args.batch_size)


        if done:
            avg_reward = episode_reward.mean().cpu().item()
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Average Reward: {avg_reward:.3f}")
            state, _ = env.reset()
            done = False
            episode_reward = torch.zeros((args.batch_size, N), device=torch.device(args.device))
            episode_timesteps = 0
            episode_num += 1

        # Evaluation
        if (t + 1) % args.eval_freq == 0:
            steps_per_sec = timer.steps_per_sec(t + 1)
            eval_len, eval_ret, _, _ = util_network.batch_eval(agent, eval_env)
            evaluations.append(eval_ret)

            if t >= args.start_timesteps:
                info.update({'eval_ret': eval_ret})

            print('Step {}. Steps per sec: {:.4g}.'.format(t + 1, steps_per_sec))

            if eval_ret > best_eval_reward:
                for i in range(agent.N):
                    best_actors[i] = agent.actors[i].state_dict()
                    best_critic_last_layer[i] = agent.critic_last_layers[i].state_dict()
                    best_critic_phi[i] = agent.phi_nets[i].state_dict()

                    # Save best models
                    torch.save(best_actors[i], log_path + f"/best_actor_{i}.pth")
                    torch.save(best_critic_last_layer[i], log_path + f"/best_critic_last_layer_{i}.pth")
                    torch.save(best_critic_phi[i], log_path + f"/best_critic_phi_{i}.pth")

            best_eval_reward = max(evaluations)

        if (t + 1) % 500 == 0 and t >= args.start_timesteps:
            for key, value in info.items():
                if 'dist' not in key:
                    summary_writer.add_scalar(f'info/{key}', value, t + 1)
                else:
                    for dist_key, dist_val in value.items():
                        summary_writer.add_histogram(dist_key, dist_val, t + 1)
            summary_writer.flush()

    summary_writer.close()

    print('Total time cost {:.4g}s.'.format(timer.time_cost()))

    # Save final models
    for i in range(N):
        torch.save(agent.actors[i].state_dict(), log_path + "/actor_last_%d.pth" %i)
        torch.save(agent.critic_last_layers[i].state_dict(), log_path + "/critic_last_layer_last_%d.pth" %i)
        torch.save(agent.phi_nets[i].state_dict(), log_path + "/critic_phi_last_%d.pth"%i)
