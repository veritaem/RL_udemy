import numpy as np
import torch as T 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from TD3actor_critic import Actor, Critic
import os

class ReplayBuffer():
    def __init__(self, size, input_shape, n_actions):
        self.mem_size=size
        self.mem_count=0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory= np.zeros((self.mem_size, n_actions))
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

class Agent():
    def __init__(self, name, fname, input_dims,env, alpha=1e-3, beta=1e-4, n_actions=2, gamma=0.99,
                tau=1, update_actor_interval=2, warmup=1000,  fc1=400, fc2=300, batch_size=100,
                noise=.1, max_size=1000000):
        self.name = name
        self.batch_size=batch_size
        self.gamma=gamma
        self.tau=tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.fname = fname
        self.learn_step=0
        self.time_step=0
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.warmup = warmup
        self.noise=noise
        self.update_actor_interval=update_actor_interval
        self.Q1 = Critic(beta, input_dims, n_actions, fc1, fc2, name='critic 1')
        self.Q2 = Critic(beta, input_dims, n_actions, fc1, fc2, name='critic 2')
        self.pi = Actor(alpha, input_dims, n_actions, fc1, fc2, name='actor')
        self.target_Q1 = Critic(beta, input_dims, n_actions, fc1, fc2, name='target critic 1')
        self.target_Q2 = Critic(beta, input_dims, n_actions, fc1, fc2, name='target critic 2')
        self.target_pi = Actor(alpha, input_dims, n_actions, fc1, fc2, name='target actor')
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

    def choose_action(self, state):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale = self.noise, size = (self.n_actions,)))
        else:
            state = T.tensor([state], dtype= T.float).to(self.pi.device)
            mu = self.pi.forward(state).to(self.pi.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise)).to(self.pi.device)
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step+=1
        return mu_prime.cpu.detach().numpy()
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_count < self.batch_size:
            return 
        state, action, reward, state_, done = \
            self.memory.sample_buffer(self.batch_size)
        state =T.tensor(state, dtype = T.float).to(self.Q1.device)
        action=T.tensor(action, dtype= T.float).to(self.Q1.device)
        reward=T.tensor(reward, dtype= T.float).to(self.Q1.device)
        state_=T.tensor(state_, dtype= T.float).to(self.Q1.device)
        done = T.tensor(done).to(self.Q1.device)

        target_actions = self.target_pi.forward(state_)
        target_actions = target_actions + \
            T.clamp(T.Tensor(np.random.normal(scale=.2)),min = -.5, max =.5 )
        target_actions = clamp(target_actions, self.min_action[0], self.max_action[0])
        q1_ = self.target_Q1.forward(state_, target_actions)
        q2_ = self.target_Q2.forward(state_, target_actions)
        q1=self.Q1.forward(state, action)
        q2=self.Q2.forward(state, action)

        q1_[done]=0.0
        q2_[done]=0.0
        q1_=q1_.view(-1)
        q2_=q2_.view(-1)
        critic_value = T.min(q1_, q2_)
        target = reward + self.gamma * critic_value
        target = target.view(self.batch_size, 1)

        self.Q1.optimizer.zero_grad()
        self.Q2.optimizer.zero_grad()
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss_total = q1_loss + q2_loss
        critic_loss_total.backward()
        self.Q1.optimizer.step()
        self.Q2.optimizer.step()
        self.learn_step+=1
        if self.learn_step % self.update_actor_interval !=0:
            return
        self.pi.optimizer.zero_grad()
        actor_q1_loss = self.Q1.forward(state, self.pi.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.pi.optimizer.step()
        self.update_network_params()

    def update_network_params(self, tau=None):
        if tau is None:
            tau=self.tau
        actor_params = self.pi.named_parameters()
        critic1_params=self.Q1.named_parameters()
        critic2_params=self.Q2.named_parameters()
        target_actor_params = self.target_pi.named_parameters()
        target_critic1_params=self.target_Q1.named_parameters()
        target_critic2_params=self.target_Q2.named_parameters()
        actor_dict= dict(actor_params)
        target_actor_dict=dict(target_actor_params)
        critic1_dict=dict(critic1_params)
        target_critic1_dict=dict(target_critic1_params)
        critic2_dict=dict(critic2_params)
        target_critic2_dict=dict(target_critic2_params)
        #update network like so : for name in params, param is equal to tau*current + 1-tau*next
        for name in actor_dict:
            actor_dict[name] = tau * actor_dict[name].clone() + (1-tau) * \
                                target_actor_dict[name].clone()
        for name in critic1_dict:
            critic1_dict[name] = tau * critic1_dict[name].clone() + (1-tau) *\
                                    target_critic1_dict[name].clone()
        for name in critic2_dict:
            critic2_dict[name] = tau * critic2_dict[name].clone() + (1-tau) *\
                                    target_critic2_dict[name].clone()
        self.target_pi.load_state_dict(actor_dict)
        self.target_Q1.load_state_dict(critic1_dict)
        self.target_Q2.load_state_dict(critic2_dict)

    def save_models(self):
        self.pi.save()
        self.target_pi.save()
        self.Q1.save()
        self.Q2.save()
        self.target_Q1.save()
        self.target_Q2.save()
    
    def load_models(self):
        self.pi.load()
        self.target_pi.load()
        self.Q1.load()
        self.target_Q1.load()
        self.Q2.load()
        self.target_Q2.load()
        






