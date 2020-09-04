import numpy as np

class OUActionNoise():
    def __init__(self, mu, sigma=.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0=x0
        self.reset()

    def __call__(self):
        x= self.x0 + self.theta * (self.mu - self.x0) * self.dt +\
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev=x

    def reset(self):
        return self.x0 if self.x0 is not None else np.zeros(self.mu)

class ReplayBuffer():
    def __init__(self, size, input_shape, n_actions):
        self.mem_size=size
        self.mem_count=0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory= np.zeroes((self.mem_size, n_actions))
        self.reward_memory= np.zeros(self.mem_size)
        self.terminal_memory= np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_count % self.mem_size
        self.state_memory[index]=state
        self.new_state_memory[index]=state_
        self.reward_memory[index]=reward
        self.action_memory[index]=action
        self.terminal_memory[index]=terminal
        self.mem_count+=1

    def sample_buffer(self, batch_size):
        max_mem =min(self.mem_count, self.mem_size)
        batch=np.random.choice(max_mem, batch_size)
        state=self.state_memory[batch]
        state_=self.new_state_memory[batch]
        action=self.action_memory[batch]
        reward=self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return state, action, reward, state_, terminal






