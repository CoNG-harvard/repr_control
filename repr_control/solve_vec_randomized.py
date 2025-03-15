import argparse
import os
import pickle as pkl

from tensorboardX import SummaryWriter
from datetime import datetime
from repr_control.utils import util, buffer
from repr_control.agent.sac import sac_agent
from repr_control.agent.rfsac import rfsac_agent
from repr_control.agent.randomized_sac import random_sac_agent
from define_problem import *
# from define_problem_linear import *
from gymnasium.envs.registration import register
import gymnasium
import yaml
from repr_control.utils.buffer import Batch
from repr_control.agent.randomized_sac import learn_phi


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### parameter that
    parser.add_argument("--alg", default="randomized_sac",
                        help="The algorithm to use. rfsac or sac or randomized_sac.")
    parser.add_argument("--env", default='custom_vec',
                        help="Name your env/dynamics, only for folder names.")  # Alg name (sac, vlsac)
    parser.add_argument("--rf_num", default=256, type=int,
                        help="Number of random features. Suitable numbers for 2-dimensional system is 512, 3-dimensional 1024, etc.")
    parser.add_argument("--rsvd_num", default=512, type=int,
                        help="Number of features for randomized svd. Suitable numbers for 2-dimensional system is 512, 3-dimensional 1024, etc.")
    parser.add_argument("--nystrom_sample_dim", default=8192, type=int,
                        help='The sampling dimension for nystrom critic. After sampling, take the maximum rf_num eigenvectors..')
    parser.add_argument("--device", default='cuda:1', type=str,
                        help="pytorch device, cuda if you have nvidia gpu and install cuda version of pytorch. "
                             "mps if you run on apple silicon, otherwise cpu.")

    ### Parameters that usually don't need to be changed.
    parser.add_argument("--seed", default=0, type=int,
                        help='random seed.')  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=0, type=float,
                        help='the number of initial steps optimizes critic_phi via random sampled actions and randomized SVD')  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=100, type=int,
                        help='number of iterations as the interval to evaluate trained policy.')  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e4, type=float,
                        help='the total training time steps / iterations.')  # Max time steps to run environment
    parser.add_argument("--batch_size", default=1024, type=int)  # Batch size for both actor and critic
    parser.add_argument("--hidden_dim", default=256, type=int)  # Network hidden dims
    parser.add_argument("--feature_dim", default=256, type=int)  # Latent feature dim
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--rsvd_sigma", default = 0.05)  # Sigma to generate random mu for rsvd
    parser.add_argument("--embedding_dim", default=-1, type=int)  # if -1, do not add embedding layer

    parser.add_argument("--use_nystrom", action='store_true')
    parser.add_argument("--use_random_feature", dest='use_nystrom', action='store_false')
    parser.set_defaults(use_nystrom=False)






    args = parser.parse_args()


    alg_name = args.alg
    exp_name = f'seed_{args.seed}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    # setup example_results
    use_V_critic = False
    log_path = f'log/{alg_name}/{env_name}/start_timesteps={args.start_timesteps}/sigma={sigma}/rsvd_sigma={args.rsvd_sigma}/V_critic={use_V_critic}/{exp_name}'
    summary_writer = SummaryWriter(log_path + "/summary_files")

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kwargs = vars(args)
    kwargs.update({
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_range": action_range,
        'obs_space_high': np.clip(state_range[0], -3., 3.).tolist(),
        'obs_space_low': np.clip(state_range[1], -3., 3.).tolist(),  # in case of inf observation space
    })


    # replay_buffer = buffer.ReplayBuffer(state_dim, action_dim, device=args.device)

    if args.env == 'custom':
        register(id='custom-v0',
                 entry_point='repr_control.envs:CustomEnv',
                 max_episode_steps=max_step)
        env = gymnasium.make('custom-v0',
                       dynamics=dynamics,
                       rewards=rewards,
                       initial_distribution = initial_distribution,
                       state_range=state_range,
                       action_range=action_range,
                       sigma=sigma)
        eval_env = gymnasium.make('custom-v0',
                            dynamics=dynamics,
                            rewards=rewards,
                            initial_distribution = initial_distribution,
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
                       initial_distribution = initial_distribution,
                       rand_distribution = rand_distribution,
                       state_range=state_range,
                       action_range=action_range,
                       sigma=sigma,
                       sample_batch_size=args.batch_size,
                       device=torch.device(args.device),)
        eval_env = CustomVecEnv(
                       dynamics=dynamics,
                       rewards=rewards,
                       initial_distribution = initial_distribution,
                       rand_distribution = rand_distribution,
                       state_range=state_range,
                       action_range=action_range,
                       sigma=sigma,
                       sample_batch_size=args.batch_size,
                       device=torch.device(args.device),)
    else:
        env = gymnasium.make(args.env)
        eval_env = gymnasium.make(args.env)
        env = gymnasium.wrappers.RescaleAction(env, min_action=-1, max_action=1)
        eval_env = gymnasium.wrappers.RescaleAction(eval_env, min_action=-1, max_action=1)

    # Evaluate untrained policy
    evaluations = []

    state, _ = env.reset()
    done = False
    episode_reward = torch.zeros((args.batch_size, 1), device=torch.device(args.device))
    episode_timesteps = 0
    episode_num = 0
    timer = util.Timer()

    # keep track of best eval model's state dict
    best_eval_reward = -1e6
    best_actor = None
    best_critic = None

    # save parameters
    # kwargs.update({"action_space": None}) # action space might not be serializable
    with open(os.path.join(log_path, 'train_params.yaml'), 'w') as fp:
        yaml.dump(kwargs, fp, default_flow_style=False)

    with open(os.path.join(log_path, 'train_params.pth'), 'wb') as f:  # 'wb' mode for writing in binary
        torch.save(kwargs, f)
        print("pytorch kwargs saved")


    # #first, train the randomized phi
    # phi_net = learn_phi.phiNet(state_dim, action_dim, hidden_dim = args.hidden_dim, output_dim = args.rsvd_num , hidden_depth  = 3).to(args.device)
    # rand_mu_net = learn_phi.randMu(state_dim, args.rsvd_num, sigma = args.rsvd_sigma).to(args.device)
    # phi_optimizer = torch.optim.Adam(phi_net.parameters(),
	# 											lr= 1e-4,
	# 											betas=[0.9, 0.999])
    


    # # env.sample_batch_size  = 100 #try reducing batch size during phi training
    # for t in range(int(args.start_timesteps)):
    #     state, _ = env.rand_reset()
    #     # print("state", state)
    #     episode_timesteps += 1
    #     action = env.sample_action()
    #     # print("action", action)
    #     next_state, reward, terminated, truncated, rollout_info = env.step(action)
    #     # print("next_state", next_state)
    #     phi = phi_net(state,action)
    #     with torch.no_grad():
    #         rand_mu = rand_mu_net(next_state)
    #     # print("phi shape", phi.shape)
    #     inner_prod = torch.sum(phi * rand_mu, dim = 1)
    #     loss_norm = torch.mean(torch.sum(phi**2,dim=1))
    #     # print("loss_norm", loss_norm)
    #     # print("inner_prod shape", inner_prod.shape)
    #     loss_self = -2 * torch.mean(inner_prod)
    #     # print("loss_self", loss_self)
    #     if t  == 0:
    #         print("inner_prod[:10], t =%d" %t, inner_prod[:10])
    #         print("loss_self, t = %d" %t, loss_self)
    #         print("loss_norm, t = %d"%t, loss_norm)
    #     loss = loss_norm + loss_self
    #     phi_optimizer.zero_grad()
    #     loss.backward()
    #     phi_optimizer.step()

    #     if t % 100 == 0:
    #         print("average inner_prod, t = %d" % t, -loss_self/2.)
    #         print("loss at t=%d" %t, loss)

    # print("phi_net computation done")

    # #Try to see if trained phi is any good, by using it as a basis, and using a mu
    # mu_net = learn_phi.muNet(state_dim, hidden_dim = args.hidden_dim, output_dim = args.rsvd_num , hidden_depth  = 3).to(args.device)
    # mu_optimizer = torch.optim.Adam(mu_net.parameters(),
	# 											lr= 3e-4,
	# 											betas=[0.9, 0.999])
    # # phi_net = learn_phi.randPhi(state_dim, action_dim,output_dim = args.rsvd_num, 
    # # dynamics_fn = dynamics, action_range = action_range,sigma = sigma, device = args.device).to(args.device) #try random fourier to see this ie better
    # print("sigma", sigma)
    # for t in range(int(1e2)):
    #     state, _ = env.rand_reset()
    #     bsize = state.size(0)
    #     action = env.sample_action()
    #     next_state, reward, terminated, truncated, rollout_info = env.step(action)
    #     noiseless_next_state = env.step_noiseless(state,action)
    #     noise = next_state - noiseless_next_state
    #     mean = torch.zeros(state_dim).to(args.device)
    #     cov = sigma**2 * torch.eye(state_dim).to(args.device)
    #     dist = torch.distributions.MultivariateNormal(mean, cov)
    #     log_probs = dist.log_prob(noise)
    #     # pdf = torch.exp(log_probs)/10. #rescale pdf to make things easier?
    #     pdf = torch.exp(log_probs)
    #     # print("pdf shape", pdf.shape)
    #     phi = phi_net(state,action)
    #     mu = mu_net(next_state)
    #     pred = torch.sum(phi * mu, dim = 1)
    #     loss_mu = torch.mean((pred - pdf)**2)
    #     mu_optimizer.zero_grad()
    #     loss_mu.backward()
    #     mu_optimizer.step()
    #     if t % 100 == 0:
    #         print("pdf, t = %d" % t, pdf)
    #         print("pred, t = %d" % t, pred)
    #         print("noise, t = %d" %t, noise)
    #         print("pred loss, t = %d" % t, loss_mu)


    # # Try a V_net

    # if use_V_critic == True:
    #     phi_net = learn_phi.randPhi_s_prime(state_dim, output_dim = args.rsvd_num, 
    #     dynamics_fn = dynamics, action_range = action_range,sigma = sigma, device = args.device).to(args.device) #try random fourier to see this ie better
    #     # phi_net = learn_phi.nnPhi_s_prime(state_dim, output_dim = args.rsvd_num, 
    #     # dynamics_fn = dynamics, action_range = action_range,sigma = sigma, device = args.device).to(args.device) #try nn to see this ie better

    env.sample_batch_size  = args.batch_size #revert back to standard batch size
    # Initialize policy using the trained phi
    if args.alg == "sac":
        agent = sac_agent.SACAgent(**kwargs)
    elif args.alg == 'rfsac':
        agent = rfsac_agent.CustomModelRFSACAgent(dynamics_fn = dynamics, rewards_fn = rewards, **kwargs)
    elif args.alg == 'randomized_sac':
        # agent = random_sac_agent.randSACAgent(dynamics_fn = dynamics, rewards_fn = rewards, critic_phi = phi_net, use_V_critic = use_V_critic,**kwargs)
        agent = random_sac_agent.randSACAgent(dynamics_fn = dynamics, rewards_fn = rewards, critic_phi = None, 
        use_V_critic = use_V_critic,sigma = sigma,**kwargs)
    else:
        raise NotImplementedError("Algorithm not implemented.")

    # next, use the phi
    state,_  = env.reset()
    for t in range(int(args.max_timesteps - args.start_timesteps)):

        episode_timesteps += 1


        action = agent.batch_select_action(state, explore=True)

        # Perform action
        next_state, reward, terminated, truncated, rollout_info = env.step(action)
        done = truncated
        # replay_buffer.add(state, action, next_state, reward, done)

        batch = Batch(
			state=state,
			action=action,
			reward=reward,
			next_state=next_state,
			done=torch.zeros(size=(state.shape[0], 1), device=torch.device(args.device)),
		)

        state = next_state.clone()
        episode_reward += reward.reshape((-1,1))
        info = {}

        info = agent.batch_train(batch)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            avg_reward = episode_reward.mean().cpu().item()
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Average Reward: {avg_reward:.3f}")
            # Reset environment
            # info.update({'ep_len': episode_timesteps})
            state, _ =  env.reset()
            done = False
            episode_reward = torch.zeros((args.batch_size, 1), device=torch.device(args.device))
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            steps_per_sec = timer.steps_per_sec(t + 1)
            eval_len, eval_ret, _, _ = util.batch_eval(agent, eval_env)
            evaluations.append(eval_ret)

            info.update({'eval_ret': eval_ret})


            print('Step {}. Steps per sec: {:.4g}.'.format(t + 1, steps_per_sec))

            if eval_ret > best_eval_reward:
                best_actor = agent.actor.state_dict()
                best_critic = agent.critic.state_dict()

                # save best actor/best critic
                torch.save(best_actor, log_path + "/best_actor.pth")
                torch.save(best_critic, log_path + "/best_critic.pth")

            best_eval_reward = max(evaluations)

        if (t + 1) % 500 == 0:
            for key, value in info.items():
                if 'dist' not in key:
                    summary_writer.add_scalar(f'info/{key}', value, t + 1)
                else:
                    for dist_key, dist_val in value.items():
                        summary_writer.add_histogram(dist_key, dist_val, t + 1)
            summary_writer.flush()

    summary_writer.close()

    print('Total time cost {:.4g}s.'.format(timer.time_cost()))

    torch.save(agent.actor.state_dict(), log_path + "/actor_last.pth")
    torch.save(agent.critic.state_dict(), log_path + "/critic_last.pth")


