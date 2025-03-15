from __future__ import annotations

from typing import Optional, Union, Tuple, Callable, SupportsFloat, Any

import numpy as np
import torch
import gymnasium
from gymnasium import spaces
from gymnasium.core import ObsType, ActType
from gymnasium.envs.registration import register

class CustomEnv(gymnasium.Env):


    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, dynamics: Callable, rewards: Callable, initial_distribution: Callable, rand_distribution: Callable,
                    state_range: list, action_range: list, sigma: float):
        self.observation_space = spaces.Box(low=np.array(state_range[0], dtype=np.float32),
                                            high=np.array(state_range[1], dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array(action_range[0], dtype=np.float32),
                                       high=np.array(action_range[1], dtype=np.float32), dtype=np.float32)
        self.dynamics = dynamics
        self.rewards = rewards
        self.initial_distribution = initial_distribution
        self.rand_distribution = rand_distribution
        self.sigma = sigma

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if options and 'state' in options.keys():
            self.state = options['state']
        else:
            self.state = self.initial_distribution(batch_size = 1).squeeze().float().numpy()

        return self.state, {}


    def rand_reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if options and 'state' in options.keys():
            self.state = options['state']
        else:
            self.state = self.rand_distribution(batch_size = 1).squeeze().float().numpy()

        return self.state, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        state = torch.from_numpy(self.state[np.newaxis, ...]).float()
        if isinstance(action, float):
            action = np.array([action])
        action = torch.from_numpy(action[np.newaxis, ...]).float()
        with torch.no_grad():
            next_state = self.dynamics(state, action)
        true_next_state = next_state.squeeze().numpy()
        noisy_next_state = true_next_state + np.random.normal(0, self.sigma, true_next_state.shape)
        self.state = np.clip(noisy_next_state, self.observation_space.low, self.observation_space.high, dtype=np.float32)
        reward = self.rewards(state, action)
        reward = reward.squeeze().item()
        done = False
        info = {}
        return self.state, reward, done, False, info

class CustomVecEnv(CustomEnv):

    def __init__(self,
                 dynamics: Callable,
                 rewards: Callable,
                 initial_distribution: Callable,
                 state_range,
                 action_range,
                 sigma,
                 rand_distribution = None,
                 sample_batch_size=1024,
                 device='cuda',
                 max_episode_steps=None):
        super().__init__(dynamics,rewards, initial_distribution,rand_distribution,state_range,action_range,sigma)
        self.sample_batch_size = sample_batch_size
        self.device = torch.device(device)
        self.obs_low = torch.tensor(state_range[0], device=self.device)
        self.obs_high = torch.tensor(state_range[1], device=self.device)
        self.action_low = torch.tensor(action_range[0], device=self.device)
        self.action_high = torch.tensor(action_range[1], device=self.device)
        self.step_counter = 0
        if max_episode_steps is None:
            self.max_episode_steps = 500
        else:
            self.max_episode_steps = max_episode_steps

    # def sample_action(self):
    #     actions = torch.rand(size=(self.sample_batch_size, self.action_low.shape[0]), device=self.device)
    #     return self.action_low.unsqueeze(dim=0) + 2 * (self.action_high - self.action_low).unsqueeze(dim=0) * actions

    #always sample_action to be between -1 and 1
    def sample_action(self):
        actions = 2 * (torch.rand(size=(self.sample_batch_size, self.action_low.shape[0]), device=self.device) - 0.5)
        return actions

    def rescale_action(self, actions):
        """
        rescale action from [-1, 1] to action range.
        """
        return self.action_low.unsqueeze(dim=0) + (self.action_high - self.action_low).unsqueeze(dim=0) * (actions + 1) * 0.5

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.step_counter = 0
        if seed:
            torch.manual_seed(seed)
        if options and 'state' in options.keys():
            self.state = options['state']
        else:
            self.state = self.initial_distribution(batch_size = self.sample_batch_size).float().to(self.device)

        return self.state, {}


    def rand_reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if options and 'state' in options.keys():
            self.state = options['state']
        else:
            self.state = self.rand_distribution(batch_size = self.sample_batch_size).float().to(self.device)

        return self.state, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:


        # print("action", action)
        # print("rescale action", self.rescale_action(action))
        reward = self.rewards(self.state, self.rescale_action(action))
        with torch.no_grad():
            next_state = self.dynamics(self.state, self.rescale_action(action))
        noisy_next_state = next_state + torch.normal(0, self.sigma, next_state.shape, device=self.device)
        # reward = self.rewards(self.state, action) #reward of current state, action
        self.state = torch.clip(noisy_next_state, self.obs_low, self.obs_high)
        # reward = reward.squeeze().item()
        done = False
        info = {}
        self.step_counter += 1
        if self.step_counter == self.max_episode_steps:
            truncated = True
        else:
            truncated = False
        return self.state, reward, done, truncated, info


    def step_noiseless(
        self, state,action):


        with torch.no_grad():
            next_state = self.dynamics(state, self.rescale_action(action))
        return next_state

def test_vec_env():
    from repr_control.define_problem import dynamics, rewards, initial_distribution, state_range, action_range, sigma
    env = CustomVecEnv(dynamics, rewards, initial_distribution, state_range, action_range, sigma)
    state, _ = env.reset()
    print(state.shape, state.device)
    action = env.sample_action()
    print(action.shape, action.device)

    print(env.reset())
    t = 0
    done = False
    while not done:
        state, reward, term, trunc,  _ = env.step(env.sample_action())
        done = trunc
        print(state.shape, state.device, reward.shape, reward.device, t)
        t += 1


def test_rescale_action():
    from repr_control.define_problem import dynamics, rewards, initial_distribution, state_range, action_range, sigma
    env = CustomVecEnv(dynamics, rewards, initial_distribution, state_range, action_range, sigma)
    state, _ = env.reset()
    action = torch.ones(env.sample_batch_size, 1).to(env.device)
    print(env.action_low, env.rescale_action(action))