import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor_critic(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1=256, fc2=256):
        super(Actor_critic, self).__init__()
        
        #same network for first two layers
        self.fc1= nn.Linear(*input_dims, fc1)
        self.fc2= nn.Linear(fc1, fc2)

        #actor output
        self.pi= nn.Linear(fc2, n_actions)

        #critic output
        self.v= nn.Linear(fc2, 1)

        self.optimizer=optim.Adam(self.parameters(), lr=lr)
        self.device=T.device('cudo:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        pi= self.pi(x)
        v= self.v(x)
        return (pi, v)

class Agent():
    def __init__(self, n_actions, input_dims, lr, fc1, fc2, gamma=0.99,):
        super(Agent, self).__init__()
        self.gamma=gamma
        self.lr=lr
        self.fc1=fc1
        self.fc2=fc2
        self.actor_critic=Actor_critic(lr, input_dims, n_actions, fc1, fc2)
        self.logprob=None

    def choose_action(self, observation):
        state=T.Tensor(observation).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities = F.softmax(probabilities, dim=1)
        categories = T.distributions.Categorical(probabilities)
        action = categories.sample()
        log_prob = categories.log_prob(action)
        self.logprob=log_prob
        return action.item()
    
    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()
        state = T.Tensor([state ]).to(self.actor_critic.device)
        state_= T.Tensor([state_]).to(self.actor_critic.device)
        reward= T.Tensor([reward]).to(self.actor_critic.device)
        _, critic_value = self.actor_critic.forward(state ) 
        _, critic_value_= self.actor_critic.forward(state_)
        #sub for returns at time t
        delta = reward + self.gamma*critic_value_* (1-int(done)) - critic_value
        actor_loss = -self.logprob*delta
        critic_loss = delta**2
        (actor_loss+critic_loss).backward()
        self.actor_critic.optimizer.step()
        
    

