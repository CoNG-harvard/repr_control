import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from repr_control.utils.util import unpack_batch
from repr_control.utils import util
from repr_control.utils.buffer import Batch
from repr_control.agent.critic import DoubleQCritic
from repr_control.agent.actor import DiagGaussianActor, DeterministicActor, StochasticActorFromDetStructureWrapper

from repr_control.agent.sac.sac_agent_network_vmap import DoubleVCritic
from repr_control.agent.randomized_sac import learn_phi


import torch.nn.init as init


#create a critic network with frozen phi and trainable weights.
class phiCritic(nn.Module):
	"""A network where only the last layer is trainable"""
	def __init__(self, phiNet,phi_last_dim = 512):
		super().__init__()
		
		# The fixed part of the network (phiNet)
		self.fixed_phi = phiNet
		# for param in self.fixed_phi.parameters():
		#     param.requires_grad = False
		self.norm = nn.LayerNorm(phi_last_dim)
		self.norm.bias.requires_grad = False

		# The trainable last layer (only weights, no bias)
		self.trainable_layer = nn.Linear(phi_last_dim, 1, bias=False)


		
		# # You can initialize the weights if necessary
		# nn.init.xavier_uniform_(self.trainable_layer.weight)

	def forward(self, obs, action):
		# Get the output from the fixed phiNet
		phi = self.fixed_phi(obs, action)
		x = self.norm(phi)
		
		# Apply the trainable last layer
		output = self.trainable_layer(x)
		
		return output, output #for simplicity of testing, just return Q1==Q2



#create a critic network with frozen phi and trainable weights.
class phiCritic_last_layer(nn.Module):
	"""A trainable last layer"""
	def __init__(self, phi_last_dim = 512):
		super().__init__()
		
		self.norm = nn.LayerNorm(phi_last_dim)
		self.norm.bias.requires_grad = False

		# The trainable last layer (only weights, no bias)
		self.trainable_layer = nn.Linear(phi_last_dim, 1, bias=False)


		
		# # You can initialize the weights if necessary
		# nn.init.xavier_uniform_(self.trainable_layer.weight)

	def forward(self, phi):
		# Get the output from the fixed phiNet
		x = self.norm(phi)
		
		# Apply the trainable last layer
		output = self.trainable_layer(x)
		
		return output, output #for simplicity of testing, just return Q1==Q2

#create a critic network with frozen phi and trainable weights. This is a V function
class phiCritic_V(nn.Module):
	"""A network where only the last layer is trainable"""
	def __init__(self, phiNet,phi_last_dim = 512):
		super().__init__()
		
		# The fixed part of the network (phiNet)
		self.fixed_phi = phiNet
		# for param in self.fixed_phi.parameters():
		#     param.requires_grad = False
		self.norm = nn.LayerNorm(phi_last_dim)
		self.norm.bias.requires_grad = False

		# The trainable last layer (only weights, no bias)
		self.trainable_layer = nn.Linear(phi_last_dim, 1, bias=False)


		
		# # You can initialize the weights if necessary
		# nn.init.xavier_uniform_(self.trainable_layer.weight)

	def forward(self, obs):
		# Get the output from the fixed phiNet
		phi = self.fixed_phi(obs)
		# x = 10 * self.norm(phi)
		x = self.norm(phi)
		
		# Apply the trainable last layer
		output = self.trainable_layer(x)
		
		return output, output #for simplicity of testing, just return Q1==Q2


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

class randSACAgent(object):
	"""
	Randomized SVD SAC agent
	"""
	def __init__(
			self, 
			state_dim, 
			action_dim, 
			action_range,
			lr=3e-4,
			discount=0.99, 
			target_update_period=2,
			tau=0.005,
			alpha=0.1,
			auto_entropy_tuning=True,
			hidden_dim=512,
			device='cpu',
			critic_phi = None,
			rewards_fn = None,
			dynamics_fn = None,
			use_V_critic = False,
			sigma = None,
			has_feature_step = True,
			**kwargs
			):

		self.steps = 0

		self.device = torch.device(device)
		self.action_range = action_range
		self.discount = discount 
		self.tau = tau 
		self.target_update_period = target_update_period
		self.learnable_temperature = auto_entropy_tuning
		self.rsvd_num = kwargs['rsvd_num']
		print("self.rsvd_num", self.rsvd_num)
		self.reward_fn = rewards_fn
		self.dynamics_fn = dynamics_fn

		self.has_feature_step = has_feature_step

		# # functions
		# print("This is use_V_critic", use_V_critic)
		# if use_V_critic == False:
		# 	self.critic = phiCritic(critic_phi, self.rsvd_num).to(self.device)
		# else:
		# 	# self.critic = phiCritic_V(critic_phi, self.rsvd_num).to(self.device)
		# 	print("I am using RFVCritic")
		# 	self.critic = RFVCritic(s_dim = state_dim, rf_num = self.rsvd_num,sigma = 1.0).to(self.device)
		# # 	self.critic = DoubleVCritic(
		# #     obs_dim=state_dim, 
		# #     hidden_dim = 256,
		# #     hidden_depth=2,
		# #     use_ortho_init=True
		# # ).to(self.device)

		# for name, param in self.critic.named_parameters():
		# 	if param.requires_grad == True:
		# 		print("param requires grad: ", name)
		self.phi_nets = [
			learn_phi.phiNet(state_dim, action_dim, 
			hidden_dim = hidden_dim, output_dim = self.rsvd_num ,
			 hidden_depth  = 2).to(self.device) for i in range(N)]
		self.rand_mu_nets = [learn_phi.randMu(state_dim, self.rsvd_num, 
		sigma = 1.0).to(self.device) for i in range(N)]
		all_phi_params = []
		for i in range(self.N):
			all_phi_params.extend(self.phi_nets[i].parameters())
		self.phi_optimizer = torch.optim.Adam(all_phi_params,
													lr= 1e-4,
													betas=[0.9, 0.999])
		self.critic_last_layers = [
			phiCritic_last_layer(self.rsvd_num).to(self.device) 
			for i in range(N)]

        self.base_phi = copy.deepcopy(self.phi_nets[0])
		self.base_phi = self.base_phi.to('meta')
		self.base_mu = copy.deepcopy(self.rand_mu_net[0])
		self.base_mu = self.base_mu.to('meta')
		self.base_critic_last_layer = copy.deepcopy(self.critic_last_layers[0])
        self.base_critic_last_layer = self.base_critic_last_layer.to('meta')

		critic_lr = kwargs['critic_lr'] if 'critic_lr' in kwargs.keys() else lr
		all_last_layer_params = []
		for i in range(self.N):
			all_last_layer_params.extend(self.critic_last_layers[i].parameters())
		self.critic_last_layer_optimizer = torch.optim.Adam(self.critic_last_layer.parameters(),
												 lr=critic_lr,
												 betas=[0.9, 0.999])

		self.critic_targets = copy.deepcopy([self.phi_nets, self.critic_last_layers])
		self.actors = [DiagGaussianActor(
			obs_dim=state_dim, 
			action_dim=action_dim,
			hidden_dim=hidden_dim,
			hidden_depth=2,
			log_std_bounds=[-5., 2.],
		).to(self.device) for i in range(self.N)]
		self.base_actor = copy.deepcopy(self.actors[0])
		self.base_actor = self.base_actor.to("meta")
        self.log_alphas = torch.tensor([np.log(alpha) for i in range(self.N)]).float().to(self.device) 
        self.log_alphas.requires_grad = True
		self.target_entropy = -action_dim
		
		 # optimizers
        all_actor_params = []
        for i in range(self.N):
            all_actor_params.extend(self.actors[i].parameters())
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
        # 										lr=lr,
        # 										betas=[0.9, 0.999])
        self.actor_optimizer = torch.optim.Adam(
            all_actor_params, lr=lr, betas=[0.9, 0.999]
        )

		self.log_alpha_optimizer = torch.optim.Adam([self.log_alphas],
													lr=lr,
													betas=[0.9, 0.999])


		# self.phi_net = learn_phi.phiNet(state_dim, action_dim, hidden_dim = hidden_dim, output_dim = self.rsvd_num , hidden_depth  = 2).to(self.device)
		# self.rand_mu_net = learn_phi.randMu(state_dim, self.rsvd_num, sigma = 1.0).to(self.device)
		# self.phi_optimizer = torch.optim.Adam(self.phi_net.parameters(),
		# 											lr= 1e-4,
		# 											betas=[0.9, 0.999])
		


    @property
    def alphas(self):
        return self.log_alphas.exp()

    def fmodel_actor(self, params, buffers, x):
        return functional_call(self.base_actor, (params, buffers), (x,))


    def fmodel_phi(self,params,buffers,x):
        return functional_call(self.base_phi,(params,buffers),(x,))

    def fmodel_mu(self,params,buffers,x):
        return functional_call(self.base_mu,(params,buffers),(x,))


    def fmodel_critic_last_layer(self,params,buffers,x):
        return functional_call(self.base_critic_last_layer,(params,buffers),(x,))

	def get_reward(self, state, action):
		reward = self.reward_fn(state, action)
		return torch.reshape(reward, (reward.shape[0], 1))

	def select_action(self, state, explore=False):
		if isinstance(state, list):
			state = np.array(state)
		state = state.astype(np.float32)
		assert len(state.shape) == 1
		state = torch.from_numpy(state).to(self.device)
		state = state.unsqueeze(0)
		dist = self.actor(state)
		action = dist.sample() if explore else dist.mean
		action = action.clamp(torch.tensor(-1, device=self.device),
							  torch.tensor(1, device=self.device))
		assert action.ndim == 2 and action.shape[0] == 1
		return util.to_np(action[0])

	def batch_select_action(self, state, explore=False):
		assert isinstance(state, torch.Tensor)
		dist = self.actor(state)
		action = dist.sample() if explore else dist.mean
		action = action.clamp(torch.tensor(-1, device=self.device),
							  torch.tensor(1, device=self.device))
		return action


	def update_target(self):
		if self.steps % self.target_update_period == 0:
			for param, target_param in zip(self.phi_net.parameters(), self.critic_target[0].parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.critic_last_layer.parameters(), self.critic_target[1].parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)				




	def critic_step(self, batch):
		"""
		Critic update step
		"""
		# state, action, reward, next_state, next_action, next_reward,next_next_state, done = unpack_batch(batch)
		state, action, next_state, reward, done = unpack_batch(batch)
	
		# for name, param in self.critic.named_parameters():
		# 	print("self critic params norm (%s)" % name, param.norm())

		with torch.no_grad():
			dist = self.actor(next_state)
			next_action = dist.rsample()
			next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)
			# next_q1, next_q2 = self.critic_target(next_state, next_action)
			# next_q1, next_q2 = self.critic_target(self.dynamics_fn(next_state, next_action))
			next_q1, next_q2 = self.critic_target[1](self.critic_target[0](next_state, next_action))
			next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_log_pi
			next_reward = self.get_reward(next_state, next_action)  # reward for new s,a
			target_q = next_reward + (1. - done) * self.discount * next_q

		q1, q2 = self.critic_last_layer(self.phi_net(state, action))
		q1_loss = F.mse_loss(target_q, q1)
		q2_loss = F.mse_loss(target_q, q2)
		q_loss = q1_loss + q2_loss

		self.critic_last_layer_optimizer.zero_grad()
		self.phi_optimizer.zero_grad()
		q_loss.backward()
		# for name, param in self.critic_last_layer.named_parameters():
		# 	print("self critic last layer params grad (%s)" % name, param.grad)
		# for name, param in self.phi_net.named_parameters():
		# 	print("self phi net params grad (%s)" % name, param.grad)
		self.critic_last_layer_optimizer.step()
		self.phi_optimizer.step()

		info = {
			'q1_loss': q1_loss.item(),
			'q2_loss': q2_loss.item(),
			'q1': q1.mean().item(),
			'q2': q2.mean().item(),
			'layer_norm_weights_norm': self.critic_last_layer.norm.weight.norm(),
		}

		dist = {
			'td_error': (torch.min(q1, q2) - target_q).cpu().detach().clone().numpy(),
			'q': torch.min(q1, q2).cpu().detach().clone().numpy()
		}

		info.update({'critic_dist': dist})

		return info


	def update_actor_and_alpha(self, batch):
		"""
		Actor update step
		"""
		# dist = self.actor(batch.state, batch.next_state)
		dist = self.actor(batch.state)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		reward = self.get_reward(batch.state, action)  # use reward in q-fn
		# q1, q2 = self.critic(batch.state, action)
		q1, q2 = self.critic_last_layer(self.phi_net(batch.state, action))
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


	def feature_step(self, batch):
		"""
		Feature update step
		"""
		# state, action, reward, next_state, next_action, next_reward,next_next_state, done = unpack_batch(batch)
		state, action, next_state, reward, done = unpack_batch(batch)
	

		phi = self.phi_net(state,action)
		with torch.no_grad():
			rand_mu = self.rand_mu_net(next_state)
		# print("phi shape", phi.shape)
		inner_prod = torch.sum(phi * rand_mu, dim = 1)
		loss_norm = torch.mean(torch.sum(phi**2,dim=1))
		# print("loss_norm", loss_norm)
		# print("inner_prod shape", inner_prod.shape)
		loss_self = -2 * torch.mean(inner_prod)
		# print("loss_self", loss_self)
		loss = loss_norm + loss_self
		# loss = loss_norm
		self.phi_optimizer.zero_grad()
		loss.backward()
		self.phi_optimizer.step()
		info = {
			'loss_norm': loss_norm.item(),
			'loss_self': loss_self.item()
		}
		return info




	def train(self, buffer, batch_size):
		"""
		One train step
		"""
		self.steps += 1

		batch = buffer.sample(batch_size)

		# Feature info
		if self.has_feature_step == True:
			feature_info = self.feature_step(batch)

		# Acritic step
		critic_info = self.critic_step(batch)

		# Actor and alpha step
		actor_info = self.update_actor_and_alpha(batch)


		# Update the frozen target models
		self.update_target()

		if self.has_feature_step == True:
			return {
				**feature_info,
				**critic_info, 
				**actor_info,
			}
		else:
			return {
				**critic_info, 
				**actor_info,
			}

	def batch_train(self, batch):

		"""
				One train step
				"""
		self.steps += 1

		#feature step
		# feature_info = self.feature_step(batch)

		# Acritic step
		critic_info = self.critic_step(batch)

		# Actor and alpha step
		actor_info = self.update_actor_and_alpha(batch)



		# Update the frozen target models
		self.update_target()

		return {
			# **feature_info,
			**critic_info,
			**actor_info,
		}