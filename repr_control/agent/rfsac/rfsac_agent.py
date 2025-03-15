import copy
import torch
from torch import nn
import torch.nn.functional as F

import torch.nn.init as init
import numpy as np
from repr_control.utils import util

from repr_control.utils.util import unpack_batch
from repr_control.agent.sac.sac_agent import SACAgent


class Critic(nn.Module):
	"""
	Critic with random fourier features
	"""

	def __init__(
			self,
			feature_dim,
			num_noise=20,
			hidden_dim=256,
	):
		super().__init__()
		self.num_noise = num_noise
		self.noise = torch.randn(
			[self.num_noise, feature_dim], requires_grad=False, device=device)

		# Q1
		self.l1 = nn.Linear(feature_dim, hidden_dim)  # random feature
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2
		self.l4 = nn.Linear(feature_dim, hidden_dim)  # random feature
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)

	def forward(self, mean, log_std):
		"""
		"""
		std = log_std.exp()
		batch_size, d = mean.shape

		x = mean[:, None, :] + std[:, None, :] * self.noise
		x = x.reshape(-1, d)

		q1 = F.elu(self.l1(x))  # F.relu(self.l1(x))
		q1 = q1.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
		q1 = F.elu(self.l2(q1))  # F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.elu(self.l4(x))  # F.relu(self.l4(x))
		q2 = q2.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
		q2 = F.elu(self.l5(q2))  # F.relu(self.l5(q2))
		# q2 = self.l3(q2) #is this wrong?
		q2 = self.l6(q2)

		return q1, q2


class RLNetwork(nn.Module):
	"""
	An abstract class for neural networks in reinforcement learning (RL). In deep RL, many algorithms
	use DP algorithms. For example, DQN uses two neural networks: a main neural network and a target neural network.
	Parameters of a main neural network is periodically copied to a target neural network. This RLNetwork has a
	method called soft_update that implements this copying.
	"""

	def __init__(self):
		super(RLNetwork, self).__init__()
		self.layers = []

	def forward(self, *x):
		return x

	def soft_update(self, target_nn: nn.Module, update_rate: float):
		"""
		Update the parameters of the neural network by
			params1 = self.parameters()
			params2 = target_nn.parameters()

			for p1, p2 in zip(params1, params2):
				new_params = update_rate * p1.data + (1. - update_rate) * p2.data
				p1.data.copy_(new_params)

		:param target_nn:   DDPGActor used as explained above
		:param update_rate: update_rate used as explained above
		"""

		params1 = self.parameters()
		params2 = target_nn.parameters()

		# bug?
		for p1, p2 in zip(params1, params2):
			new_params = update_rate * p1.data + (1. - update_rate) * p2.data
			p1.data.copy_(new_params)

	def train(self, loss, optimizer):
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


#currently hardcoding s_dim
#this is a  V function
class RFVCritic(RLNetwork):
	def __init__(self, s_dim=3, embedding_dim=-1, rf_num=256, sigma=0.0, learn_rf=False, **kwargs):
		super().__init__()
		self.n_layers = 1
		self.feature_dim = rf_num

		self.sigma = sigma
		# self.sigma = 0.05 #testing
		print("self.sigma", self.sigma)

		print("embedding dim", embedding_dim)
		if embedding_dim != -1:
			self.embed = nn.Linear(s_dim, embedding_dim)
		else:  # we don't add embed in this case
			embedding_dim = s_dim
			self.embed = nn.Linear(s_dim, s_dim)
			init.eye_(self.embed.weight)
			init.zeros_(self.embed.bias)
			self.embed.weight.requires_grad = False
			self.embed.bias.requires_grad = False

		# fourier_feats1 = nn.Linear(sa_dim, n_neurons)
		fourier_feats1 = nn.Linear(embedding_dim, self.feature_dim)
		# fourier_feats1 = nn.Linear(s_dim,n_neurons)
		if self.sigma > 0:
			init.normal_(fourier_feats1.weight, std=1. / self.sigma)
			# pass
		else:
			init.normal_(fourier_feats1.weight)
		init.uniform_(fourier_feats1.bias, 0, 2 * np.pi)
		# init.zeros_(fourier_feats.bias)
		fourier_feats1.weight.requires_grad = learn_rf
		fourier_feats1.bias.requires_grad = learn_rf
		self.fourier1 = fourier_feats1  # unnormalized, no cosine/sine yet

		fourier_feats2 = nn.Linear(embedding_dim, self.feature_dim)
		# fourier_feats2 = nn.Linear(s_dim,n_neurons)
		if self.sigma > 0:
			init.normal_(fourier_feats2.weight, std=1. / self.sigma)
			# pass
		else:
			init.normal_(fourier_feats2.weight)
		init.uniform_(fourier_feats2.bias, 0, 2 * np.pi)
		fourier_feats2.weight.requires_grad = learn_rf
		fourier_feats2.bias.requires_grad = learn_rf
		self.fourier2 = fourier_feats2

		layer1 = nn.Linear(self.feature_dim, 1)  # try default scaling
		# init.uniform_(layer1.weight, -3e-3,3e-3) #weight is the only thing we update
		init.zeros_(layer1.bias)
		layer1.bias.requires_grad = False  # weight is the only thing we update
		self.output1 = layer1

		layer2 = nn.Linear(self.feature_dim, 1)  # try default scaling
		# init.uniform_(layer2.weight, -3e-3,3e-3) 
		# init.uniform_(layer2.weight, -3e-4,3e-4)
		init.zeros_(layer2.bias)
		layer2.bias.requires_grad = False  # weight is the only thing we update
		self.output2 = layer2

		self.norm1 = nn.LayerNorm(self.feature_dim)
		self.norm1.bias.requires_grad = False

		self.norm = nn.LayerNorm(self.feature_dim)
		self.norm.bias.requires_grad = False

	def forward(self, states: torch.Tensor):
		x = states
		# print("x initial norm",torch.linalg.norm(x))
		# x = torch.cat([states,actions],axis = -1)
		# x = F.batch_norm(x) #perform batch normalization (or is dbn better?)
		# x = (x - torch.mean(x, dim=0))/torch.std(x, dim=0) #normalization
		# x = self.bn(x)
		x = self.embed(x) #use an embedding layer
		# print("x embedding norm", torch.linalg.norm(x))
		# x = F.relu(x)
		x1 = self.fourier1(x)
		x2 = self.fourier2(x)
		x1 = torch.cos(x1)
		x2 = torch.cos(x2)
		# x1 = torch.cos(x)
		# x2 = torch.sin(x)
		# x = torch.cat([x1,x2],axis = -1)
		# x = torch.div(x,1./np.sqrt(2 * self.feature_dim))
		# if self.sigma > 0:
		#   x1 = torch.multiply(x1,1./np.sqrt(2 * np.pi * self.sigma))
		#   x2 = torch.multiply(x2,1./np.sqrt(2 * np.pi * self.sigma)) 
		# x1 = torch.div(x1,np.sqrt(self.feature_dim/2))
		# x2 = torch.div(x2,np.sqrt(self.feature_dim/2))
		# x1 = torch.div(x1,1./self.feature_dim)
		# x2 = torch.div(x2,1./self.feature_dim)
		# change to layer norm
		# x1 = 10. * self.norm1(x1)
		# x2 = 10. * self.norm(x2)
		x1 = self.norm1(x1)
		x2 = self.norm(x2)
		# print("x1 norm", torch.linalg.norm(x1,axis = 1))
		# x = torch.relu(x)
		return self.output1(x1), self.output2(x2)
		# return self.output1(x1),self.output1(x1) #just testing if the min actually helps

	def get_norm(self):
		l1_norm = torch.norm(self.output1)
		l2_norm = torch.norm(self.output2)
		return (l1_norm, l2_norm)

class nystromVCritic(RLNetwork):
	def __init__(self,
				 s_dim=3,
				 s_low=np.array([-1, -1, -8]),
				 feat_num=256,
				 sigma=0.0,
				 buffer=None,
				 learn_rf=False,
				 **kwargs):
		super().__init__()
		self.n_layers = 1
		self.feature_dim = feat_num
		self.sigma = sigma
		# s_high = -s_low

		self.s_low = kwargs.get('obs_space_low')
		self.s_high = kwargs.get('obs_space_high')
		self.s_dim = kwargs.get('state_dim')
		self.s_dim = self.s_dim[0] if (not isinstance(self.s_dim, int)) else self.s_dim
		# self.feature_dim = kwargs.get('random_feature_dim')
		self.sample_dim = kwargs.get('nystrom_sample_dim')
		# self.sigma = kwargs.get('sigma')
		self.dynamics_type = kwargs.get('dynamics_type')
		self.sin_input = kwargs.get('dynamics_parameters').get('sin_input')
		self.dynamics_parameters = kwargs.get('dynamics_parameters')

		eval = kwargs.get('eval', False)
		if not eval:
			np.random.seed(kwargs.get('seed'))
			# create nystrom feats
			self.nystrom_samples1 = np.random.uniform(self.s_low, self.s_high, size=(self.sample_dim, self.s_dim))
			# self.nystrom_samples1 = np.random.uniform([-0.3, -0.03, 0.3, -0.03, 0.955, 0., -0.03],
			#                                           [0.3, 0.03, 0.7, 0.03, 1., 0.295, 0.03], size=(self.sample_dim, self.s_dim))
			# self.nystrom_samples2 = np.random.uniform(s_low,s_high,size = (feat_num, s_dim))

			if sigma > 0.0:
				self.kernel = lambda z: np.exp(-np.linalg.norm(z) ** 2 / (2. * sigma ** 2))
			else:
				self.kernel = lambda z: np.exp(-np.linalg.norm(z) ** 2 / (2.))
			K_m1 = self.make_K(self.nystrom_samples1, self.kernel)
			print('start eig')

			[eig_vals1, S1] = np.linalg.eig(
				K_m1)  # numpy linalg eig doesn't produce negative eigenvalues... (unlike torch)

			# truncate top k eigens
			argsort = np.argsort(eig_vals1)[::-1]
			eig_vals1 = eig_vals1[argsort]
			S1 = S1[:, argsort]
			eig_vals1 = np.clip(eig_vals1, 1e-8, np.inf)[:self.feature_dim]
			self.eig_vals1 = torch.from_numpy(eig_vals1).float().to(device)
			self.S1 = torch.from_numpy(S1[:, :self.feature_dim]).float().to(device)
			self.nystrom_samples1 = torch.from_numpy(self.nystrom_samples1).to(device)
		else:
			self.nystrom_samples1 = torch.zeros((self.sample_dim, self.s_dim))
			self.eig_vals1 = torch.ones([self.feature_dim, ])
			self.S1 = torch.zeros([self.s_dim, self.feature_dim])

		layer1 = nn.Linear(self.feature_dim, 1)  # try default scaling
		init.zeros_(layer1.bias)
		layer1.bias.requires_grad = False  # weight is the only thing we update
		self.output1 = layer1

		layer2 = nn.Linear(self.feature_dim, 1)  # try default scaling
		init.zeros_(layer2.bias)
		layer2.bias.requires_grad = False  # weight is the only thing we update
		self.output2 = layer2

		self.norm = nn.LayerNorm(self.feature_dim)
		self.norm.bias.requires_grad = False

	def make_K(self, samples, kernel):
		print('start cal K')
		m, d = samples.shape
		K_m = np.empty((m, m))
		for i in np.arange(m):
			for j in np.arange(m):
				K_m[i, j] = kernel(samples[i, :] - samples[j, :])
		return K_m

	def kernel_matrix_numpy(self, x1, x2):
		print('start cal K')
		dx2 = np.expand_dims(x1, axis=1) - np.expand_dims(x2,
														  axis=0)  # will return the kernel matrix of k(x1, x2) with symmetric kernel.
		if self.sigma > 0.0:
			K_x2 = np.exp(-np.linalg.norm(dx2, axis=2) ** 2 / (2. * self.sigma ** 2))
		else:
			K_x2 = np.exp(-np.linalg.norm(dx2, axis=2) ** 2 / (2.))
		return K_x2

	def forward(self, states: torch.Tensor):
		x1 = self.nystrom_samples1.unsqueeze(0) - states.unsqueeze(1)
		K_x1 = torch.exp(-torch.linalg.norm(x1, axis=2) ** 2 / 2).float()
		phi_all1 = (K_x1 @ (self.S1)) @ torch.diag((self.eig_vals1.clone() + 1e-8) ** (-0.5))
		# phi_all1 = self.norm(phi_all1)
		phi_all1 = 50. * phi_all1
		phi_all1 = phi_all1.to(torch.float32)
		return self.output1(phi_all1), self.output2(phi_all1)

	def get_norm(self):
		l1_norm = torch.norm(self.output1)
		l2_norm = torch.norm(self.output2)
		return (l1_norm, l2_norm)


class DoubleQCritic(nn.Module):
	"""Critic network, employes double Q-learning."""

	def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
		super().__init__()

		self.Q1 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
		self.Q2 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

		self.outputs = dict()
		self.apply(util.weight_init)

	def forward(self, obs, action):
		assert obs.size(0) == action.size(0)

		obs_action = torch.cat([obs, action], dim=-1)
		q1 = self.Q1(obs_action)
		q2 = self.Q2(obs_action)

		self.outputs['q1'] = q1
		self.outputs['q2'] = q2

		return q1, q2

class CustomModelRFSACAgent(SACAgent):

	def __init__(self,
				 state_dim,
				 action_dim,
				 action_range,
				 dynamics_fn,
				 rewards_fn,
				 lr=3e-4,
				 discount=0.99,
				 target_update_period=2,
				 tau=0.005,
				 alpha=0.1,
				 auto_entropy_tuning=True,
				 hidden_dim=256,
				 sigma=0.0,
				 rf_num=256,
				 learn_rf=False,
				 use_nystrom=False,
				 replay_buffer=None,
				 device = 'cpu',
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

		self.sigma = sigma
		if use_nystrom == False:  # use RF
			self.critic = RFVCritic(s_dim=state_dim, sigma=sigma, rf_num=rf_num, learn_rf=learn_rf, **kwargs).to(self.device)
		else:  # use nystrom
			feat_num = rf_num
			self.critic = nystromVCritic(sigma=sigma, feat_num=feat_num, buffer=replay_buffer, learn_rf=learn_rf,
										 **kwargs).to(self.device)
		# self.critic = Critic().to(device)
		for name, param in self.critic.named_parameters():
			if param.requires_grad == True:
				print("param requires grad: ", name)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=lr, betas=[0.9, 0.999])
		self.args = kwargs
		self.dynamics = dynamics_fn
		self.reward_fn = rewards_fn

	def get_reward(self, state, action):
		reward = self.reward_fn(state, action)
		return torch.reshape(reward, (reward.shape[0], 1))

	def update_actor_and_alpha(self, batch):
		"""
		Actor update step
		"""
		# dist = self.actor(batch.state, batch.next_state)
		dist = self.actor(batch.state)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		reward = self.get_reward(batch.state, action)  # use reward in q-fn
		q1, q2 = self.critic(self.dynamics(batch.state, action))
		q = self.discount * torch.min(q1, q2) + reward

		actor_loss = ((self.alpha) * log_prob - q).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		info = {'actor_loss': actor_loss.item()}

		if self.learnable_temperature:
			self.log_alpha_optimizer.zero_grad()
			alpha_loss = (self.alpha *
						  (-log_prob - self.target_entropy).detach()).mean()
			alpha_loss.backward()
			self.log_alpha_optimizer.step()

			info['alpha_loss'] = alpha_loss
			info['alpha'] = self.alpha

		return info

	def critic_step(self, batch):
		"""
		Critic update step
		"""
		# state, action, reward, next_state, next_action, next_reward,next_next_state, done = unpack_batch(batch)
		state, action, next_state, reward, done = unpack_batch(batch)
		# for name, param in self.critic.named_parameters():
		# 	print("self critic params norm (%s)" % name, param.norm())
		# 	print("sigma", self.sigma)

		with torch.no_grad():
			dist = self.actor(next_state)
			next_action = dist.rsample()
			next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)
			next_q1, next_q2 = self.critic_target(self.dynamics(next_state, next_action))
			next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_log_pi
			next_reward = self.get_reward(next_state, next_action)  # reward for new s,a
			target_q = next_reward + (1. - done) * self.discount * next_q

		q1, q2 = self.critic(self.dynamics(state, action))
		q1_loss = F.mse_loss(target_q, q1)
		q2_loss = F.mse_loss(target_q, q2)
		q_loss = q1_loss + q2_loss

		self.critic_optimizer.zero_grad()
		q_loss.backward()
		self.critic_optimizer.step()

		info = {
			'q1_loss': q1_loss.item(),
			'q2_loss': q2_loss.item(),
			'q1': q1.mean().item(),
			'q2': q2.mean().item(),
			'layer_norm_weights_norm': self.critic.norm.weight.norm(),
		}

		dist = {
			'td_error': (torch.min(q1, q2) - target_q).cpu().detach().clone().numpy(),
			'q': torch.min(q1, q2).cpu().detach().clone().numpy()
		}

		info.update({'critic_dist': dist})

		return info

	def train(self, buffer, batch_size):
		"""
		One train step
		"""
		self.steps += 1

		batch = buffer.sample(batch_size)

		# Acritic step
		critic_info = self.critic_step(batch)
		# critic_info = self.rfQcritic_step(batch)

		# Actor and alpha step
		actor_info = self.update_actor_and_alpha(batch)

		# Update the frozen target models
		self.update_target()

		return {
			**critic_info,
			**actor_info,
		}

	
