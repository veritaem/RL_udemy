import numpy as np
import torch as T 
import os
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

class Critic(nn.Module):
    def __init__(self, beta, input_dims, fc1, fc2, n_actions, name, fname='tmp/critic'):
        super(Critic, self).__init__()
        self.input_dims=input_dims
        self.fc1dims=fc1
        self.fc2dims=fc2
        self.n_actions=n_actions
        self.name=name
        self.file_name = os.path.join(self.fname, name+'_ddpg')

        self.fc1=nn.Linear(*input_dims, self.fc1dims)
        self.fc2=nn.Linear(self.fc1dims, self.fc2dims)
        self.norm1=nn.LayerNorm(self.fc1dims)
        self.norm2=nn.LayerNorm(self.fc2dims)

        self.action_value= nn.Linear(self.n_actions, self.fc2dims)
        self.q=nn.Linear(self.fc2dims, 1)

        f1 = 1./np.sqrt(self.fc1dims.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2dims.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = .003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.dat.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer=optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.norm1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.norm2(state_value)
        state_value = F.relu(state_value)
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        return state_action_value

    def save_checkpoint(self):
        print('...saving...')
        T.save(self.state_dict(), self.file_name)
        print(f'save to {self.file_name} complete!')

    def load_checkpoint(self):
        print(f'...now loading from {self.file_name}...')
        self.load_state_dict(T.load(self.file_name))
        print('loaded!')

class Actor(nn.Module):
    def __init__(self, name, input_shape,  alpha, fc1_dims, fc2_dims, n_actions,
                fname='tmp/actor'):
        super(Actor, self).__init__()
        self.input_shape=input_shape
        self.n_actions=n_actions
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.name=name
        self.checkpoint = os.path.join(fname, name+'_ddpg')

        #layers
        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims) 
        self.norm1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.norm2 = nn.LayerNorm(self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        #weights
        f1=1./np.sqrt(self.fc1_dims.weight.data.shape()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2=1./np.sqrt(self.fc2_dims.weight.data.shape()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3=.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer=optim.Adam(self.parameters(), lr=alpha,
                                weight_decay=.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.norm1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.norm2(state_value)
        state_value = F.relu(state_value)
        state_value = F.tanh(self.mu(state_value))
        return state_value

    def save_checkpoint(self):
        print('...saving...')
        T.save(self.state_dict(), self.checkpoint)
        print(f'saved! to {self.checkpoint}')

    def load_checkpoint(self):
        print(f'...loading from {self.checkpoint}...')
        self.load_state_dict(T.load(self.checkpoint))
        print('loaded!')

    

