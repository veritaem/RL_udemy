import numpy as np
import torch as T 
import torch.nn.functional as F 
import torch.nn as nn
import torch.optim as optim
import os

class Actor(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, fc1_dims, fc2_dims, name='actor', fname='tmp/TD3'):
        super(Actor, self).__init__()
        self.name = name
        self.alpha = alpha
        self.input_dims=input_dims
        self.n_actions=n_actions
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.file_loc = os.path.join(fname, name+'TD3 Actor') 
        self.create_network()

    def create_network(self):
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(params=self.parameters(), lr=self.alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(self.device, self.name)
        self.to(self.device)

    def forward(self, state):
        state = T.tensor([state], dtype=np.float64).to(self.device)
        x= self.fc1(state)
        x= T.relu(x)
        x= self.fc2(x)
        x= T.relu(x)
        action = F.tanh(self.pi(x))
        return action

    def save(self):
        print('...saving...')
        T.save(self.state_dict(), self.file_loc)
        print('saved!')
    def load(self):
        print('loading....')
        self.load_state_dict(T.load(self.file_loc))
        print('loaded!')








class Critic(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims, fc2_dims, name='critic', fname='tmp/TD3'):
        super(Critic, self).__init__()
        self.name = name
        self.beta = beta
        self.input_dims=input_dims
        self.n_actions=n_actions
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.file_loc = os.path.join(fname, name+'TD3 Critic') 
        self.create_network()

    def create_network(self):
        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.Q = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(self.device, self.name)
        self.to(self.device)

    def forward(self, state, action):
        x=self.fc1(T.cat([state, action], dim=1))
        x=T.relu(x)
        x=self.fc2(x)
        x=T.relu(x)
        x=self.Q(x)
        return x

    def save(self):
        print('...saving critic...')
        T.save(self.state_dict(), self.file_loc)
        print('saved critic!')
    def load(self):
        print('...loading critic...')
        self.load_state_dict(T.load(self.file_loc))
        print('loaded critic!')
