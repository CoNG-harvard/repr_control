import copy
import torch
from torch import nn
import torch.nn.functional as F

import torch.nn.init as init
import numpy as np
from repr_control.utils import util_network, util

from repr_control.utils.util import unpack_batch
from repr_control.agent.sac.sac_agent_network_vmap import SACAgent, DoubleVCritic

from repr_control.agent.actor_network_vmap import (
	DiagGaussianActor,
	DeterministicActor,
	StochasticActorFromDetStructureWrapper,
	SquashedNormal,
)


from torch.func import stack_module_state, functional_call
from torch import vmap

from torch.distributions import Normal
import math




# Define PhiNet, which will replace learn_phi.phiNet
class predNet(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256, hidden_depth=2):
		super().__init__()
		self.model = util.mlp(
			input_dim=state_dim + action_dim,
			hidden_dim=hidden_dim,
			output_dim=state_dim,
			hidden_depth=hidden_depth
		)
	
	def forward(self, state, action):
		# print("action shape", action.shape)
		x = torch.cat([state, action], dim=-1)
		# print("x shape", x.shape)
		pred_next = self.model(x)
		return pred_next


# Define PhiNet, which will replace learn_phi.phiNet
class PhiNet(nn.Module):
	def __init__(self, state_dim, action_dim, output_dim=512, hidden_dim=256, hidden_depth=2):
		super().__init__()
		self.model = util.mlp(
			input_dim=state_dim + action_dim,
			hidden_dim=hidden_dim,
			output_dim=output_dim,
			hidden_depth=hidden_depth
		)
	
	def forward(self, state, action):
		# print("action shape", action.shape)
		x = torch.cat([state, action], dim=-1)
		# print("x shape", x.shape)
		phi = self.model(x)
		return phi


# PhiNet that takes in only s.
class PhiNet_s_only(nn.Module):
	def __init__(self, state_dim, output_dim=512, hidden_dim=256, hidden_depth=2):
		super().__init__()
		self.model = util.mlp(
			input_dim=state_dim,
			hidden_dim=hidden_dim,
			output_dim=output_dim,
			hidden_depth=hidden_depth
		)
	
	def forward(self, state):
		phi = self.model(state)
		return phi

# Define RandMu, which will replace learn_phi.randMu
class RandMu(nn.Module):
	def __init__(self, state_dim, output_dim=512, sigma=1.0):
		super().__init__()
		self.model = nn.Linear(state_dim, output_dim, bias=False)
		nn.init.normal_(self.model.weight, mean=0.0, std=1./sigma)
		for param in self.model.parameters():
			param.requires_grad = False  # Make it non-trainable
	
	def forward(self, state):
		return self.model(state)

# Create a critic network with frozen phi and trainable weights.
class PhiCriticLastLayer(nn.Module):
	"""A trainable last layer"""
	def __init__(self, phi_last_dim=512):
		super().__init__()
		self.norm = nn.LayerNorm(phi_last_dim)
		self.norm.bias.requires_grad = False
		# The trainable last layer (only weights, no bias)
		self.trainable_layer = nn.Linear(phi_last_dim, 1, bias=False)

	def forward(self, phi):
		# Apply LayerNorm
		x = self.norm(phi)
		# x = phi
		# Apply the trainable last layer
		output = self.trainable_layer(x)
		return output, output  # For simplicity, return Q1 == Q2

class CustomPhiSACAgent(SACAgent):
	"""
	Randomized SVD SAC agent rewritten in the style of CustomModelSACAgent.
	"""
	def __init__(
			self, 
			state_dim, 
			action_dim, 
			action_range,
			dynamics_fn,
			rewards_fn,
			lr=3e-4,
			# lr = 1e-4,
			discount=0.99, 
			target_update_period=2,
			tau=0.005,
			alpha=0.1,
			auto_entropy_tuning=True,
			hidden_dim=256,
			device='cpu',
			rsvd_num=256,
			critic_lr=None,
			N=1,  # Number of agents for vmap
			eval_adjacency = None,
			policy_adjacency = None,
			eval_minus_one_adjacency = None,
			kappa_obs_dim = 1,
			eval_kappa_obs_dim = 1,
			eval_kappa_action_dim = 1,
			critic_use_ortho_init = True,
			**kwargs
			):

		super().__init__(
			state_dim=state_dim,
			action_dim=action_dim,
			action_range=action_range,
			lr=lr,
			tau=tau,
			alpha=alpha,
			discount=discount,
			target_update_period=target_update_period,
			auto_entropy_tuning=auto_entropy_tuning,
			hidden_dim=hidden_dim,
			device=device,
			**kwargs
		)

		self.has_feature_step = kwargs['has_feature_step']
		print("has feat step", kwargs['has_feature_step'])

		# print("eval_kappa_obs_dim", eval_kappa_obs_dim)

		print("lr", lr)
		self.lr = lr

		self.N = N  # Number of copies for vmap
		self.device = torch.device(device)
		self.policy_adjacency = policy_adjacency
		self.eval_adjacency = eval_adjacency
		self.eval_minus_one_adjacency = eval_minus_one_adjacency

		self.action_low = torch.tensor(action_range[0], device=self.device)
		self.action_high = torch.tensor(action_range[1], device=self.device)
		self.dynamics = dynamics_fn
		self.reward_fn = rewards_fn

		# Initialize phi_nets and rand_mu_nets for vmap
		self.phi_nets = [PhiNet(eval_kappa_obs_dim, eval_kappa_action_dim, 
		output_dim=rsvd_num, hidden_dim=hidden_dim).to(self.device) for _ in range(self.N)]
		# self.pred_nets = [predNet(eval_kappa_obs_dim, eval_kappa_action_dim, 
		# hidden_dim=hidden_dim).to(self.device) for _ in range(self.N)]
		# self.phi_nets = [PhiNet_s_only(eval_kappa_obs_dim,
		# output_dim=rsvd_num, hidden_dim=hidden_dim).to(self.device) for _ in range(self.N)]
		eval_kappa_minus_one_obs_dim = max(0,eval_kappa_obs_dim - 2*state_dim)
		# print("eval kappa minus one obs dim", eval_kappa_minus_one_obs_dim)
		self.rand_mu_nets = [RandMu(eval_kappa_minus_one_obs_dim, output_dim=rsvd_num, sigma=1.0).to(self.device) for _ in range(self.N)]

		# # Optimizer for pred_nets
		# all_pred_params = []
		# for pred_net in self.pred_nets:
		# 	all_pred_params.extend(pred_net.parameters())
		# self.pred_optimizer = torch.optim.Adam(
		# 	all_pred_params,
		# 	lr=self.lr,
		# 	betas=[0.9, 0.999]
		# )


		# Optimizer for phi_nets
		all_phi_params = []
		for phi_net in self.phi_nets:
			all_phi_params.extend(phi_net.parameters())
		self.phi_optimizer = torch.optim.Adam(
			all_phi_params,
			lr=self.lr,
			betas=[0.9, 0.999]
		)
		# self.base_pred = copy.deepcopy(self.pred_nets[0])
		# self.base_pred = self.base_pred.to('meta')
		self.base_phi = copy.deepcopy(self.phi_nets[0])
		self.base_phi = self.base_phi.to('meta')
		self.base_mu = copy.deepcopy(self.rand_mu_nets[0])
		self.base_mu = self.base_mu.to('meta')


		# Initialize critic_last_layers
		self.critic_last_layers = [PhiCriticLastLayer(phi_last_dim=rsvd_num).to(self.device) for _ in range(self.N)]
		all_critic_params = []
		for critic in self.critic_last_layers:
			all_critic_params.extend(critic.parameters())
		critic_lr = critic_lr if critic_lr is not None else lr
		self.critic_last_layer_optimizer = torch.optim.Adam(
			all_critic_params,
			lr=critic_lr,
			betas=[0.9, 0.999]
		)
		self.base_critic_last_layer = copy.deepcopy(self.critic_last_layers[0])
		self.base_critic_last_layer = self.base_critic_last_layer.to('meta')

		# Actor networks for each agent
		print("eval_kappa_obs_dim", eval_kappa_obs_dim)
		self.actors = [DiagGaussianActor(
			obs_dim=kappa_obs_dim, 
			action_dim=action_dim,
			hidden_dim=hidden_dim,
			# hidden_depth=2,
			hidden_depth = 3,
			log_std_bounds=[-5., 2.],
		).to(self.device) for _ in range(self.N)]

		self.base_actor = copy.deepcopy(self.actors[0])
		self.base_actor = self.base_actor.to("meta")

		# Log alpha for entropy regularization
		self.log_alphas = torch.tensor([np.log(alpha) for i in range(self.N)]).float().to(self.device) 
		self.log_alphas.requires_grad = True
		print("self log alpha", self.log_alphas)
		self.target_entropy = -action_dim

		# Optimizers
		all_actor_params = []
		for actor in self.actors:
			all_actor_params.extend(actor.parameters())
		self.actor_optimizer = torch.optim.Adam(
			all_actor_params,
			lr=lr,
			betas=[0.9, 0.999]
		)
		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alphas], lr=lr, betas=[0.9, 0.999]
		)

		# Critic target networks
		self.phi_nets_targets = copy.deepcopy(self.phi_nets)
		self.critic_last_layers_targets = copy.deepcopy(self.critic_last_layers)

	@property
	def alphas(self):
		return self.log_alphas.exp()

	def fmodel_actor(self, params, buffers, x):
		return functional_call(self.base_actor, (params, buffers), (x,))


	def fmodel_phi(self,params,buffers,x,y):
		return functional_call(self.base_phi,(params,buffers),(x,y))

	def fmodel_pred(self,params,buffers,x,y):
		return functional_call(self.base_pred,(params,buffers),(x,y))

	def fmodel_mu(self,params,buffers,x):
		return functional_call(self.base_mu,(params,buffers),(x,))


	def fmodel_critic_last_layer(self,params,buffers,x):
		return functional_call(self.base_critic_last_layer,(params,buffers),(x,))


	def rescale_action(self, actions):
		"""
		rescale action from [-1, 1] to action range.
		"""
		result = self.action_low.unsqueeze(dim=0) + (self.action_high - self.action_low).unsqueeze(dim=0) * (actions + 1) * 0.5
		return result.to(torch.float32) #make everything float32

	# Outputs a B by N by (2*kappa + 1) *m tensor
	def get_local_states_actions(self, states: torch.Tensor, actions:torch.Tensor):
		# first reshape states into B by N by m
		# print("states", states)
		batchsize = states.size()[0]
		states = torch.reshape(states, (batchsize, self.N, -1))
		local_states_concat = torch.reshape(
			states[:, self.eval_adjacency, :], (batchsize, self.N, -1)
		)
		actions = torch.reshape(actions, (batchsize, self.N, -1))
		local_actions_concat = torch.reshape(
			actions[:, self.eval_adjacency, :], (batchsize, self.N, -1)
		)
		# print("local) actions concat", local_actions_concat)
		return local_states_concat, local_actions_concat
	# Outputs a B by N by (2*kappa + 1) *m tensor
	def get_local_states(self, states: torch.Tensor):
		# first reshape states into B by N by m
		# print("states", states)
		batchsize = states.size()[0]
		states = torch.reshape(states, (batchsize, self.N, -1))
		local_states_concat = torch.reshape(
			states[:, self.policy_adjacency, :], (batchsize, self.N, -1)
		)
		# print("Loca; shape", local_states_concat.shape)
		# print("N", self.N)
		return local_states_concat


	def get_local_states_kappa_minus_one(self, states: torch.Tensor):
		# first reshape states into B by N by m
		# print("states", states)
		batchsize = states.size()[0]
		states = torch.reshape(states, (batchsize, self.N, -1))
		local_states_concat = torch.reshape(
			states[:, self.eval_minus_one_adjacency, :], (batchsize, self.N, -1)
		)
		# print("self.eval_minus_one_adjacency", self.eval_minus_one_adjacency.shape)
		# print("Loca; shape", local_states_concat.shape)
		# print("N", self.N)
		return local_states_concat


	def get_reward(self, state, action):
		reward = self.reward_fn(state, action)
		# return torch.reshape(reward, (reward.shape[0], 1))
		return torch.reshape(reward, (reward.shape[0], -1))

	def select_action(self, state, explore=False):
		if isinstance(state, np.ndarray):
			state = torch.from_numpy(state).to(self.device).float()
		state = state.unsqueeze(0)
		dist = self.actors[0](state)  # Use the first actor for action selection
		action = dist.sample() if explore else dist.mean
		action = action.clamp(-1, 1)
		return action[0].cpu().numpy()

	def batch_select_action_network(self, states, explore=False):
		assert isinstance(states, torch.Tensor)
		self.explore  = explore
		params, buffers = stack_module_state(self.actors)
		local_states_concat = self.get_local_states(states)
		# print("local syas shape", local_states_concat.shape)
		# print("local states concat shape", local_states_concat.shape)
		mu_vmap, std_vmap = vmap(self.fmodel_actor, in_dims=(0, 0, 1), out_dims=1)(
			params, buffers, local_states_concat
		)
		actions_vmap = vmap(self.batch_select_action_local, randomness="different", in_dims=(1, 1), out_dims=1)(
			mu_vmap, std_vmap
		)
		# print("success with vmapping sampling!")
		action = torch.reshape(actions_vmap, (states.size()[0], -1))
		# dist = actor(local_states)
		# action = dist.sample() if explore else dist.mean
		# action = action.clamp(torch.tensor(-1, device=self.device),
		# 					  torch.tensor(1, device=self.device))
		# print("action", action)
		return action

	def update_target(self):
		if self.steps % self.target_update_period == 0:
			for i in range(self.N):
				for param, target_param in zip(self.phi_nets[i].parameters(), self.phi_nets_targets[i].parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				for param, target_param in zip(self.critic_last_layers[i].parameters(), self.critic_last_layers_targets[i].parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def critic_step(self, batch):
		"""
		Critic update step
		"""
		# state, action, reward, next_state, next_action, next_reward,next_next_state, done = unpack_batch(batch)
		state, action, next_state, reward, done = unpack_batch(batch)


		# pred_params, pred_buffers = stack_module_state(self.pred_nets) 
		phi_params, phi_buffers = stack_module_state(self.phi_nets) #don't no grad this!!
		critic_last_layer_params, critic_last_layer_buffers = stack_module_state(self.critic_last_layers)
		with torch.no_grad():
			phi_target_params, phi_target_buffers = stack_module_state(self.phi_nets_targets) #don't no grad this!!
			critic_last_layer_target_params, critic_last_layer_target_buffers = stack_module_state(self.critic_last_layers_targets)
			next_state = batch.next_state
			local_next_states_concat = self.get_local_states(next_state)
			actor_params, actor_buffers = stack_module_state(self.actors)
			# critic_params, critic_buffers = stack_module_state(self.critics)
			mu_vmap, std_vmap = vmap(self.fmodel_actor, in_dims = (0,0,1),out_dims = 1)(actor_params,actor_buffers,local_next_states_concat)
			noise = torch.randn(mu_vmap.shape).to(self.device)
			next_actions_vmap_og = mu_vmap + std_vmap * noise
			next_actions_vmap = torch.tanh(next_actions_vmap_og)
			# next_actions_vmap = vmap(self.batch_select_action_local, randomness = "different",
			#     in_dims = (1,1),out_dims=1)(mu_vmap,std_vmap)
			norm_dist = Normal(loc = 0., scale = 1.)
			next_log_prob_vmap = norm_dist.log_prob(noise) - 2. * (math.log(2.) - next_actions_vmap_og - F.softplus(-2. * next_actions_vmap_og))
			# next_log_prob_vmap = vmap(self.batch_get_log_prob_local, in_dims = (1,1,1),out_dims = 1)(mu_vmap,std_vmap,next_actions_vmap)
			next_action = torch.reshape(next_actions_vmap,(next_state.size(0),-1))
			# next_next_state_noiseless_concat = self.get_local_states(self.dynamics(batch.next_state,next_action))
			# next_next_state_noiseless_concat = self.get_local_states(self.dynamics(batch.next_state,self.rescale_action(next_action)))
			local_next_states_concat, local_next_actions_concat = self.get_local_states_actions(next_state, next_action)

			# next_pred = vmap(self.fmodel_pred, in_dims = (0,0,1,1), out_dims = 1)(pred_params, pred_buffers,local_next_states_concat,local_next_actions_concat)
			next_phi = vmap(self.fmodel_phi, in_dims = (0,0,1,1), out_dims = 1)(phi_target_params,phi_target_buffers, local_next_states_concat,local_next_actions_concat)
			# next_phi = vmap(self.fmodel_phi, in_dims = (0,0,1), out_dims = 1)(phi_target_params,phi_target_buffers, next_next_state_noiseless_concat)
			next_q1, next_q2 = vmap(self.fmodel_critic_last_layer, in_dims = (0,0,1), out_dims = 1)(critic_last_layer_target_params, 
			critic_last_layer_target_buffers,next_phi)
			# next_q = vmap(self.fmodel_critic, in_dims = (0,0,1),out_dims = 1)(critic_target_params,
			# critic_target_buffers,next_next_state_noiseless_concat)
			# next_q1 = torch.squeeze(next_q1)
			# next_q2 = torch.squeeze(next_q2)
			next_q1 = torch.reshape(next_q1, (next_q1.shape[0],self.N))
			next_q2 = torch.reshape(next_q2, (next_q2.shape[0],self.N))    
			next_action_log_pi = torch.reshape(next_log_prob_vmap, (batch.next_state.size(0),-1))
			# print("self.alphs shape", self.alphas.shape)
			# print("next action log pi shape", next_action_log_pi.shape)
			# print("next q1 shape", next_q1.shape)
			# next_q = self.discount * torch.min(next_q1,next_q2) - self.alphas/self.N * next_action_log_pi
			next_q = self.discount * torch.min(next_q1,next_q2) - self.alphas * next_action_log_pi

			# dist = self.actor(next_state)
			# next_action = dist.rsample()
			# next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)
			# next_q1, next_q2 = self.critic_target(self.dynamics(next_state, next_action))
			# next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_log_pi
			next_reward = self.get_reward(next_state, self.rescale_action(next_action))  # reward for new s,a
			target_q = next_reward + (1. - done) * self.discount * next_q

		phi_params_need_grad_keys = []
		for key in phi_params.keys():
			if phi_params[key].requires_grad == True:
				phi_params_need_grad_keys += [key]
	
		critic_last_layer_params_need_grad_keys = []
		for key in critic_last_layer_params.keys():
			if critic_last_layer_params[key].requires_grad == True:
				critic_last_layer_params_need_grad_keys += [key]

		local_states_concat, local_actions_concat = self.get_local_states_actions(state, action)

		new_phi_optimizer = torch.optim.Adam(
			[phi_params[key] for key in phi_params_need_grad_keys], lr=self.lr, betas=[0.9, 0.999])

		new_critic_last_layer_optimizer = torch.optim.Adam(
			[critic_last_layer_params[key] for key in critic_last_layer_params_need_grad_keys], lr=self.lr, betas=[0.9, 0.999])

		# print("this is new_critic_optimizer", new_critic_optimizer)
		with torch.no_grad():
			next_state_noiseless_concat = self.get_local_states(self.dynamics(state,self.rescale_action(action)))

		# next_state_noiseless_concat = self.get_local_states(self.dynamics(state,action))
		# pred = vmap(self.fmodel_pred, in_dims = (0,0,1,1), out_dims = 1)(pred_params,pred_buffers, 
		# local_states_concat, local_actions_concat)

		phi = vmap(self.fmodel_phi, in_dims = (0,0,1,1), out_dims = 1)(phi_params,phi_buffers,
		local_states_concat, local_actions_concat)
		# phi = vmap(self.fmodel_phi, in_dims = (0,0,1), out_dims = 1)(phi_params,phi_buffers,
		#  next_state_noiseless_concat)
		q1, q2 = vmap(self.fmodel_critic_last_layer, in_dims = (0,0,1),out_dims = 1)(critic_last_layer_params,
			critic_last_layer_buffers,phi)
		# q1 = torch.squeeze(q1)
		# q2 = torch.squeeze(q2)
		q1 = torch.reshape(q1, (q1.shape[0],self.N))
		q2 = torch.reshape(q2, (q2.shape[0],self.N))

		q1_loss = F.mse_loss(target_q, q1)
		q2_loss = F.mse_loss(target_q, q2)
		q_loss = q1_loss + q2_loss

		self.phi_optimizer.zero_grad()
		self.critic_last_layer_optimizer.zero_grad()
		new_phi_optimizer.zero_grad()
		new_critic_last_layer_optimizer.zero_grad()
		# self.critics[0].output1.weight.retain_grad()
		# q1.retain_grad()
		q_loss.backward()

		for i in range(self.N):
			for name, param in self.phi_nets[i].named_parameters():
				if param.requires_grad == True:
					# print("this is new critic param name: %s grad" %name, phi_params[name].grad[0])
					# print("this is new param name: %s tensor" %name, critic_params[name][i])
					param.grad = phi_params[name].grad[i].clone()
					# print("param grad norm", param.grad.norm())
					# param = param + 1.0 * param.grad
					# print(f"updated Parameter name: {name}, size: {param.size()}")
			for name, param in self.critic_last_layers[i].named_parameters():
				if param.requires_grad == True:
					# print("this is new critic param name: %s grad" %name, critic_last_layer_params[name].grad[0])
					# print("this is new param name: %s tensor" %name, critic_params[name][i])
					param.grad = critic_last_layer_params[name].grad[i].clone()
					# param.data = torch.zeros(size=param.data.size()).to(self.device)
					# print("after update, norm", param.norm())
		self.phi_optimizer.step()
		self.critic_last_layer_optimizer.step()
		# print("self critic state", self.critic_optimizer.state_dict())



		# for i in range(self.N):
		# 	for name, param in self.phi_nets[i].named_parameters():
		# 		if param.requires_grad == True:
		# 			if name == "model.2.weight":
		# 				print("this is param[:10], key = %s, after critic step" %name, param[:10])

		info = {
			'q1_loss': q1_loss.item(),
			'q2_loss': q2_loss.item(),
			'q1': q1.mean().item(),
			'q2': q2.mean().item(),
		}

		#maybe sth wrong this code
		# print("q1 shape", q1.shape)
		dist = {
			'td_error (0)': (torch.min(q1[:,0], q2[:,0]) - target_q[:,0]).cpu().detach().clone().numpy(),
			'q (0)': torch.min(q1[:,0], q2[:,0]).cpu().detach().clone().numpy()
		}

		info.update({'critic_dist': dist})

		return info





	def update_actor_and_alpha(self, batch):


		obs = batch.state
		action = batch.action
		# local_states_concat, local_actions_concat = self.get_local_states_actions(obs,action)
		local_states_concat= self.get_local_states(obs)

		actor_params, actor_buffers = stack_module_state(self.actors)

		actor_params_need_grad_keys = []
		for key in actor_params.keys():
			if actor_params[key].requires_grad == True:
				actor_params_need_grad_keys += [key]

		new_actor_optimizer = torch.optim.Adam(
			[actor_params[key] for key in actor_params_need_grad_keys], lr=self.lr, betas=[0.9, 0.999])

		# Stack parameters for vmap
		# actor_params, actor_buffers = stack_module_state(self.actors)
		# phi_nets_params, phi_nets_buffers = stack_module_state(self.phi_nets)
		# critic_params, critic_buffers = stack_module_state(self.critic_last_layers)
		
		mu_vmap, std_vmap = vmap(self.fmodel_actor,
			in_dims = (0,0,1),out_dims = 1)(actor_params,actor_buffers,local_states_concat)
		noise = torch.randn(mu_vmap.shape).to(self.device)
		actions_vmap_og = mu_vmap + std_vmap * noise
		actions_vmap = torch.tanh(actions_vmap_og)
		norm_dist = Normal(loc = 0., scale = 1.)
		# this is a more numerically stable version of log abs determinant for tanh
		# see https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
		log_prob_vmap = norm_dist.log_prob(noise) - 2. * (math.log(2.) - actions_vmap_og - F.softplus(-2. * actions_vmap_og))
		log_prob_vmap = torch.reshape(log_prob_vmap, (log_prob_vmap.shape[0],self.N))
		action = torch.reshape(actions_vmap,(obs.size(0),-1))
		reward = self.get_reward(batch.state, action)  # use reward in q-fn
		# reward = self.get_reward(batch.state, self.rescale_action(action))  # use reward in q-fn
		
		# next_state_noiseless_concat = self.get_local_states(self.dynamics(batch.state,action))
		# next_state_noiseless_concat = self.get_local_states(self.dynamics(batch.state,self.rescale_action(action)))
		# pred_params, pred_buffers = stack_module_state(self.pred_nets)
		phi_params, phi_buffers = stack_module_state(self.phi_nets)
		for name, param in self.critic_last_layers[0].named_parameters():
			if param.requires_grad == True:
				# print("this is new critic param name: %s grad" %name, critic_last_layer_params[name].grad[0])
				# print("this is new param name: %s tensor" %name, critic_params[name][i])
				# print("in actor, param norm", param.norm())
				pass

		local_states_concat, local_actions_concat = self.get_local_states_actions(obs,action)
		critic_last_layer_params, critic_last_layer_buffers = stack_module_state(self.critic_last_layers)
		# pred = vmap(self.fmodel_pred, in_dims = (0,0,1,1), out_dims = 1)(pred_params, pred_buffers,
		# local_states_concat, local_actions_concat)
		phi_vmap= vmap(self.fmodel_phi, in_dims = (0,0,1,1),out_dims = 1)(phi_params,
			phi_buffers, local_states_concat,local_actions_concat)
		# phi_vmap= vmap(self.fmodel_phi, in_dims = (0,0,1),out_dims = 1)(phi_params,
		# 	phi_buffers, next_state_noiseless_concat)
		# print("phi_vmap", phi_vmap)
		q1_vmap, q2_vmap = vmap(self.fmodel_critic_last_layer, in_dims = (0,0,1), out_dims =1)(
			critic_last_layer_params, critic_last_layer_buffers, phi_vmap
		)



		q1_vmap = torch.reshape(q1_vmap, (q1_vmap.shape[0],self.N))
		q2_vmap = torch.reshape(q2_vmap,(q2_vmap.shape[0],self.N))

		q = self.discount * torch.min(q1_vmap,q2_vmap) + reward


		# actor_losses = (self.alphas/self.N * log_prob_vmap - q).mean(dim = 0) 
		actor_losses = (self.alphas * log_prob_vmap -  q).mean(dim = 0) 
		actor_loss = torch.sum(actor_losses) #sum up the N (avg) losses

		if self.steps %50 ==0:
			print("q1_vmap norm at step=%d" % self.steps, q1_vmap.norm())
			print("actor loss", actor_loss)
			print("mu_vmap norm", mu_vmap.norm())


		self.actor_optimizer.zero_grad()
		new_actor_optimizer.zero_grad()
		actor_loss.backward()

		for i in range(self.N):
			for name, param in self.actors[i].named_parameters():
				if param.requires_grad == True:
					param.grad = actor_params[name].grad[i].clone()
				if self.steps %50 ==0 and name == "trunk.0.weight":
					print("param = %s, grad first entries at step=%d" % (name,self.steps), param.grad[:2])
		self.actor_optimizer.step()



		info = {'actor_loss': actor_loss.item()}
		# print("actor loss", actor_loss)

		if self.learnable_temperature:
			self.log_alpha_optimizer.zero_grad()
			alpha_losses = (self.alphas *
						  (-log_prob_vmap - self.target_entropy).detach()).mean(dim = 0)
			alpha_loss = torch.sum(alpha_losses)
			alpha_loss.backward()
			self.log_alpha_optimizer.step()
			alphas_updated = self.alphas
			info['alpha_losses 1'] = alpha_losses[0]
			info['alpha 1'] = alphas_updated[0] #this is the alphas before the update

		return info

	def feature_step(self, batch):


		state, action, next_state, reward, done = unpack_batch(batch)

		local_states_concat, local_actions_concat = self.get_local_states_actions(state,action)
		local_next_states_concat = self.get_local_states_kappa_minus_one(next_state)


		# print("local next concat", local_next_states_concat.shape)
		# Stack parameters for vmap
		phi_nets_params, phi_nets_buffers = stack_module_state(self.phi_nets)
		rand_mu_params, rand_mu_buffers = stack_module_state(self.rand_mu_nets)

		phi_params_need_grad_keys = []
		for key in phi_nets_params.keys():
			if phi_nets_params[key].requires_grad == True:
				phi_params_need_grad_keys += [key]


		new_phi_optimizer = torch.optim.Adam(
			[phi_nets_params[key] for key in phi_params_need_grad_keys], lr=self.lr, betas=[0.9, 0.999])

		# print("local states shape", local_states_concat.shape)
		# print("local actions shape", local_actions_concat.shape)
		# Compute phi and rand_mu using vmap
		phi = vmap(self.fmodel_phi, in_dims=(0, 0, 1, 1), out_dims = 1)(phi_nets_params, phi_nets_buffers, local_states_concat, local_actions_concat)
		rand_mu = vmap(self.fmodel_mu, in_dims=(0, 0, 1), out_dims = 1)(rand_mu_params, rand_mu_buffers, local_next_states_concat)

		# Compute feature loss
		inner_prod = (phi * rand_mu).sum(dim=-1)
		loss_norm = (phi ** 2).sum(dim=-1).mean(dim = 0)
		# print("inner_prod shape", inner_prod.shape)
		# print("loss norm shape", loss_norm.shape)
		loss_self = -2 * inner_prod.mean(dim = 0)
		loss_norm = loss_norm.sum()
		loss_self = loss_self.sum()
		# loss = loss_norm.sum() + loss_self.sum()
		loss = loss_norm + loss_self
		# loss = loss_self
		# loss = loss_norm

		# Optimize phi_nets
		self.phi_optimizer.zero_grad()
		new_phi_optimizer.zero_grad()
		loss.backward()


		for i in range(self.N):
			for name, param in self.phi_nets[i].named_parameters():
				if param.requires_grad == True:
					param.grad = phi_nets_params[name].grad[i].clone()
					# print("Im here", param.grad)
					# print("key = %s, param norm:%.5f" %(name, param.norm()))

		self.phi_optimizer.step()


		# for i in range(self.N):
		# 	for name, param in self.phi_nets[i].named_parameters():
		# 		if param.requires_grad == True:
		# 			if name == "model.2.weight":
		# 				print("key = %s, param[:10] after feat step: " %(name), param[:10])


		info = {
			# 'loss_norm': loss_norm.sum().item(),
			# 'loss_self': loss_self.sum().item()
			'loss_norm': loss_norm.item(),
			'loss_self': loss_self.item()
		}

		return info



	def pred_step(self, batch):


		state, action, next_state, reward, done = unpack_batch(batch)

		local_states_concat, local_actions_concat = self.get_local_states_actions(state,action)
		local_next_states_concat = self.get_local_states(next_state)

		# Stack parameters for vmap
		pred_nets_params, pred_nets_buffers = stack_module_state(self.pred_nets)

		pred_params_need_grad_keys = []
		for key in pred_nets_params.keys():
			if pred_nets_params[key].requires_grad == True:
				pred_params_need_grad_keys += [key]


		new_pred_optimizer = torch.optim.Adam(
			[pred_nets_params[key] for key in pred_params_need_grad_keys], lr=self.lr, betas=[0.9, 0.999])

		# print("local states shape", local_states_concat.shape)
		# print("local actions shape", local_actions_concat.shape)
		# Compute phi and rand_mu using vmap
		pred = vmap(self.fmodel_pred, in_dims=(0, 0, 1, 1), out_dims = 1)(pred_nets_params, pred_nets_buffers, local_states_concat, local_actions_concat)

		# Compute feature loss
		# next_state_noiseless_concat = self.get_local_states(self.dynamics(batch.state,self.rescale_action(action)))
		loss = F.mse_loss(pred, local_next_states_concat)
		# loss = F.mse_loss(local_states_concat, local_next_states_concat)

		# print("local_next_states_concat first 5", local_next_states_concat[:5])
		# print("pred - local_next first 5", (pred - local_next_states_concat)[:5])

		# loss = loss_self
		# loss = loss_norm

		# Optimize phi_nets
		self.pred_optimizer.zero_grad()
		new_pred_optimizer.zero_grad()
		loss.backward()


		for i in range(self.N):
			for name, param in self.pred_nets[i].named_parameters():
				if param.requires_grad == True:
					param.grad = pred_nets_params[name].grad[i].clone()
					# print("Im here", param.grad)
					# print("key = %s, param norm:%.5f" %(name, param.norm()))

		self.pred_optimizer.step()


		# for i in range(self.N):
		# 	for name, param in self.phi_nets[i].named_parameters():
		# 		if param.requires_grad == True:
		# 			if name == "model.2.weight":
		# 				print("key = %s, param[:10] after feat step: " %(name), param[:10])


		info = {
			# 'loss_norm': loss_norm.sum().item(),
			# 'loss_self': loss_self.sum().item()
			'loss_pred': loss.item(),
			# 'loss_self': loss_self.item()
		}

		return info

	def batch_train(self, batch):
		"""
		One train step
		"""
		self.steps += 1

		# # Feature info
		if self.has_feature_step == True:
			# print("i here")
			feature_info = self.feature_step(batch)

		# pred_info = self.pred_step(batch)

		# Critic step
		critic_info = self.critic_step(batch)

		# # Critic step
		# critic_info = self.critic_step(batch)

		# # Critic step
		# critic_info = self.critic_step(batch)



		# Actor and alpha step
		actor_info = self.update_actor_and_alpha(batch)

		# Update the frozen target models
		self.update_target()

		if self.has_feature_step == True:
			return {
				**feature_info,
				# **pred_info,
				**critic_info,
				**actor_info,
			}
		else:
			return {
				**critic_info,
				**actor_info,
			}
	def train(self, buffer, batch_size):
		"""
		One train step
		"""
		self.steps += 1

		batch = buffer.sample(batch_size)

		# # Feature info
		if self.has_feature_step == True:
			# print("i here")
			feature_info = self.feature_step(batch)

		# pred_info = self.pred_step(batch)

		# Critic step
		critic_info = self.critic_step(batch)

		# # Critic step
		# critic_info = self.critic_step(batch)

		# # Critic step
		# critic_info = self.critic_step(batch)



		# Actor and alpha step
		actor_info = self.update_actor_and_alpha(batch)


		# Update the frozen target models
		self.update_target()

		return {
			**critic_info,
			**actor_info,
		}