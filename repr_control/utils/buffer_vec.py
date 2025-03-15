import collections
import torch

Batch = collections.namedtuple(
    'Batch',
    # Include any additional fields you need
    ['state', 'action', 'reward', 'next_state', 'done']
)

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, N = 1,max_size=int(1e6), device='cpu'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)
        self.N = N

        # Initialize buffer arrays as PyTorch tensors
        self.state = torch.zeros((max_size, state_dim * N), device=self.device)
        self.action = torch.zeros((max_size, action_dim * N), device=self.device)
        self.reward = torch.zeros((max_size, N), device=self.device)
        self.next_state = torch.zeros((max_size, state_dim *N), device=self.device)
        self.done = torch.zeros((max_size, N), device=self.device)

    def add(self, state, action, next_state, reward, done):
        # Convert inputs to torch tensors if they're not
        state = torch.as_tensor(state, device=self.device)
        action = torch.as_tensor(action, device=self.device)
        reward = torch.as_tensor(reward, device=self.device).reshape(-1, self.N)  # Ensure correct shape
        next_state = torch.as_tensor(next_state, device=self.device)



        batch_size = state.shape[0]

        # Calculate indices where new data will be inserted
        indices = torch.arange(self.ptr, self.ptr + batch_size) % self.max_size

        # Insert data into buffer arrays
        self.state[indices] = state
        self.action[indices] = action
        self.reward[indices] = reward
        self.next_state[indices] = next_state
        self.done[indices] = done #done is actually just a scalar

        # Update pointer and size
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)

        return Batch(
            state=self.state[ind],
            action=self.action[ind],
            reward=self.reward[ind],
            next_state=self.next_state[ind],
            done=self.done[ind]
        )
