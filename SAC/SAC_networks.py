import torch as T 
import torch.nn as nn 
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import os

class Critic(nn.Module):
    def __init__(self, beta, input_dims, fc1=400, fc2=300, n_actions=2, name='critic', fname= 'tmp/SAC_critic'):
        super(Critic, self).__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.name = name
        self.fname = fname
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1
        self.fc2_dims = fc2
        self.n_actions = n_actions
        self.file = os.path.join(self.fname, self.name + '_SAC')

        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr = self.beta)
        self.to(self.device)

    def forward(self, state, action):
        sa = T.cat([state, action], dim=1)
        x= self.fc1(sa)
        x= T.relu(x)
        x = self.fc2(x)
        x=T.relu(x)
        critic_value = self.q(x)
        return critic_value

    def save(self):
        T.save(self.state_dict(), self.file)
    def load(self):
        self.load_state_dict(T.load(self.file))

class Actor(nn.Module):
    def __init__(self, max_action, alpha, input_dims, fc1=400, fc2=300, n_actions=2, name='actor', 
                fname= 'tmp/SAC_actor'):
        super(Actor, self).__init__()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.name = name
        self.fname = fname
        self.alpha = alpha
        self.input_dims = input_dims
        self.fc1_dims = fc1
        self.fc2_dims = fc2
        self.n_actions = n_actions
        self.max_action = max_action
        self.reparam_noise = 1e-6
        self.file = os.path.join(self.fname, self.name + '_SAC')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr = self.alpha)
        self.to(self.device)

    def forward(self, state):
        sa = T.tensor(state, dtype=T.float).to(self.device)
        x= self.fc1(sa)
        x= T.relu(x)
        x = self.fc2(x)
        x=T.relu(x)
        mu_value = self.mu(x)
        sigma_value = self.sigma(x)
        sigma_value = T.clamp(sigma_value, min=self.reparam_noise, max=1)
        return mu_value, sigma_value

    def sample_normal(self, state, reparamaterize = True):
        mu, sigma = self.forward(state)
        probabilities=Normal(mu, sigma)
        if reparamaterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        actions = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs-=T.log(1-actions.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        return actions, log_probs

    def save(self):
        T.save(self.state_dict(), self.file)
    def load(self):
        self.load_state_dict(T.load(self.file))

class Value_Network(nn.Module):
    def __init__(self, input_dims, fc1, fc2, beta, name='value', fname = 'tmp/value'):
        super(Value_Network, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1
        self.fc2_dims = fc2
        self.beta = beta
        self.name = name
        self.fname = fname
        self.file = os.path.join(self.fname, self.name+'_SAC')

        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.value = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state= T.tensor(state, dtype=T.float).to(self.device)
        x=self.fc1(state)
        x=T.relu(x)
        x=self.fc2(x)
        x=T.relu(x)
        x=self.value(x)

        return x

    def save(self):
        T.save(self.state_dict(), self.file)
    def load(self):
        self.load_state_dict(T.load(self.file))

    