import collections
import torch

Batch = collections.namedtuple(
    'Batch',
    # Include any additional fields you need
    ['state', 'obs','action', 'reward', 'next_state', 'next_obs', 'done']
)

class ReplayBuffer(object):
    def __init__(self, state_dim, obs_dim, action_dim, max_size=int(1e6), device='cpu'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)

        # Initialize buffer arrays as PyTorch tensors
        self.state = torch.zeros((max_size, state_dim), device=self.device)
        self.obs = torch.zeros((max_size, obs_dim), device=self.device)
        self.action = torch.zeros((max_size, action_dim), device=self.device)
        self.reward = torch.zeros((max_size, 1), device=self.device)
        self.next_state = torch.zeros((max_size, state_dim), device=self.device)
        self.next_obs = torch.zeros((max_size, obs_dim), device=self.device)
        self.done = torch.zeros((max_size, 1), device=self.device)

    def add(self, state, obs,action, next_state, next_obs,reward, done):
        # Convert inputs to torch tensors if they're not
        state = torch.as_tensor(state, device=self.device)
        obs = torch.as_tensor(obs, device=self.device)
        action = torch.as_tensor(action, device=self.device)
        reward = torch.as_tensor(reward, device=self.device).reshape(-1, 1)  # Ensure correct shape
        next_state = torch.as_tensor(next_state, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)

        batch_size = state.shape[0]

        # Calculate indices where new data will be inserted
        indices = torch.arange(self.ptr, self.ptr + batch_size) % self.max_size

        # Insert data into buffer arrays
        self.state[indices] = state
        self.obs[indices] = obs
        self.action[indices] = action
        self.reward[indices] = reward
        self.next_state[indices] = next_state
        self.next_obs[indices] = next_obs
        self.done[indices] = done #done is actually just a scalar

        # Update pointer and size
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)

        return Batch(
            state=self.state[ind],
            obs=self.obs[ind],
            action=self.action[ind],
            reward=self.reward[ind],
            next_state=self.next_state[ind],
            next_obs=self.next_obs[ind],
            done=self.done[ind]
        )
