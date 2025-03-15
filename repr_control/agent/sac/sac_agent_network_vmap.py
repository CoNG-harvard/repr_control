import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from repr_control.utils import util_network
from repr_control.utils.buffer import Batch
from repr_control.agent.critic import DoubleVCritic, DoubleQCritic
from repr_control.agent.actor_network_vmap import (
    DiagGaussianActor,
    DeterministicActor,
    StochasticActorFromDetStructureWrapper,
    SquashedNormal,
)


from torch.func import stack_module_state
from torch.func import functional_call
import copy
from torch import vmap


class SACAgent(object):
    """
    DDPG Agent
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
        hidden_dim=1024,
        device="cpu",
        N=1,
        policy_adjacency=None,
        kappa_obs_dim = 1,
        **kwargs,
    ):
        self.steps = 0
        self.N = N

        self.device = torch.device(device)
        self.action_range = action_range
        self.discount = discount
        self.tau = tau
        self.target_update_period = target_update_period
        self.learnable_temperature = auto_entropy_tuning

        # functions
        self.critic = DoubleQCritic(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=2,
        ).to(self.device)
        self.critic_target = DoubleQCritic(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=2,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actors = [
            DiagGaussianActor(
                # obs_dim=state_dim,
                obs_dim = kappa_obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                # hidden_depth=2,
                hidden_depth = 3,
                log_std_bounds=[-5.0, 2.0],
            ).to(self.device)
            for i in range(self.N)
        ]
        print("actor 0")
        print(self.actors[0])
        print("state dim", state_dim)
        print("action dim", action_dim)
        print("kapps obs dim", kappa_obs_dim)
        self.base_model = copy.deepcopy(self.actors[0])
        self.base_model = self.base_model.to("meta")
        # self.log_alphas = [
        #     torch.tensor(np.log(alpha)).float().to(self.device) for i in range(self.N)
        # ]
        self.log_alphas = torch.tensor([np.log(alpha) for i in range(self.N)]).float().to(self.device) 
        self.log_alphas.requires_grad = True
        # self.log_alphas = torch.tensor([np.log(alpha) for i in range(self.N)], device = self.device, requires_grad = True) 
        # print("is leaf?", self.log_alphas[0].is_leaf)
        # for i in range(self.N):
        #     self.log_alphas[i].requires_grad = True
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
        # critic_lr = kwargs["critic_lr"] if "critic_lr" in kwargs.keys() else lr
        # self.critic_optimizer = torch.optim.Adam(
        #     self.critic.parameters(), lr=critic_lr, betas=[0.9, 0.999]
        # )

        # all_log_alpha_params = []
        # for i in range(self.N):
        # 	all_log_alpha_params.extend(self.log_alphas[i].parameters)
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alphas], lr=lr, betas=[0.9, 0.999]
        )

    # @property
    # def alpha(self):
    #     return self.log_alpha.exp()


    @property
    def alphas(self):
        return self.log_alphas.exp()

    def fmodel_actor(self, params, buffers, x):
        return functional_call(self.base_model, (params, buffers), (x,))

    def select_action(self, state, explore=False):
        if isinstance(state, list):
            state = np.array(state)
        state = state.astype(np.float32)
        assert len(state.shape) == 1
        state = torch.from_numpy(state).to(self.device)
        state = state.unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if explore else dist.mean
        action = action.clamp(
            torch.tensor(-1, device=self.device), torch.tensor(1, device=self.device)
        )
        assert action.ndim == 2 and action.shape[0] == 1
        return util.to_np(action[0])

    def batch_select_action(self, state, explore=False):
        assert isinstance(state, torch.Tensor)
        dist = self.actor(state)
        action = dist.sample() if explore else dist.mean
        action = action.clamp(
            torch.tensor(-1, device=self.device), torch.tensor(1, device=self.device)
        )
        return action

    #always explore, to avoid if/else vmap compatibility issues
    def batch_select_action_local(self,mu, std):
        # assert isinstance(local_states, torch.Tensor)
        # explore = self.explore
        dist = SquashedNormal(mu, std)
        # action = dist.sample() if explore else dist.mean
        action = dist.sample()
        # action = action.clamp(
        #     torch.tensor(-0.99, device=self.device), torch.tensor(0.99, device=self.device)
        # ) 
        return action

    def batch_get_log_prob_local(self,mu, std, action):
        dist = SquashedNormal(mu, std)
        # print("hey I was able to create dist")
        temp = dist.log_prob(action)
        # print("temp here")
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    # Takes in a B by N * m tensor
    # Outputs a B by N by (2*kappa + 1) *m tensor
    def get_local_states(self, states: torch.Tensor):
        # first reshape states into B by N by m
        # print("states", states)
        batchsize = states.size()[0]
        states = torch.reshape(states, (batchsize, self.N, -1))
        local_states_concat = torch.reshape(
            states[:, self.policy_adjacency, :], (batchsize, self.N, -1)
        )
        # print("local_states_concat", local_states_concat)
        return local_states_concat


    # Takes in a B by N * m tensor
    # Outputs a B by N by (2*kappa + 1) *m tensor
    def get_local_states_critic(self, states: torch.Tensor):
        # first reshape states into B by N by m
        # print("states", states)
        batchsize = states.size()[0]
        states = torch.reshape(states, (batchsize, self.N, -1))
        local_states_concat = torch.reshape(
            states[:, self.eval_adjacency, :], (batchsize, self.N, -1)
        )
        # print("local_states_concat", local_states_concat)
        return local_states_concat

    # output is B by (N * a_dim)
    def batch_select_action_network(self, states, explore=False):
        assert isinstance(states, torch.Tensor)
        self.explore  = explore
        params, buffers = stack_module_state(self.actors)
        local_states_concat = self.get_local_states(states)
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
        return action

    def update_target(self):
        if self.steps % self.target_update_period == 0:
            for i in range(self.N):
                for param, target_param in zip(
                    self.critics[i].parameters(), self.critic_targets[i].parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

    def critic_step(self, batch):
        """
        Critic update step
        """
        obs, action, next_obs, reward, done = util.unpack_batch(batch)
        not_done = 1.0 - done

        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": critic_loss.item(),
            "q1": current_Q1.mean().item(),
            "q2": current_Q1.mean().item(),
        }

    def update_actor_and_alpha(self, batch):
        obs = batch.state
        local_states_concat = self.get_local_states(obs)
        params, buffers = stack_module_state(self.actors)
        mu_vmap, std_vmap = vmap(fmodel, in_dims=(0, 0, 1), out_dims=1)(
            params, buffers, local_states_concat
        )
        action_vmap = vmap(batch_select_action_local, in_dims=(1, 1), out_dims=1)(
            mu_vmap, std_vmap
        )
        log_prob_vmap = vmap(batch_get_log_prob_local, in_dims=(1, 1, 1), out_dims=1)(
            mu_vmap, std_vmap, action_vmap
        )
        action = torch.reshape(actions_vmap, (states.size()[0], -1))

        # dist = self.actor(obs)
        # action = dist.rsample()
        # log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        info = {"actor_loss": actor_loss.item()}

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            info["alpha_loss"] = alpha_loss
            info["alpha"] = self.alpha

        return info

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1

        batch = buffer.sample(batch_size)
        # Acritic step
        critic_info = self.critic_step(batch)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch)

        # Update the frozen target models
        self.update_target()

        return {
            **critic_info,
            **actor_info,
        }

    def batch_train(self, batch):
        """
        One train step
        """
        self.steps += 1

        # Acritic step
        critic_info = self.critic_step(batch)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch)

        # Update the frozen target models
        self.update_target()

        return {
            **critic_info,
            **actor_info,
        }
