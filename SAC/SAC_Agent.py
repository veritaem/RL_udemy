import numpy as np
from SAC_networks import Actor, Value_Network, Critic
import torch as T 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
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
    def __init__(self, alpha, beta, tau, input_dims, n_actions, env, env_id, gamma=0.99,
                max_size =1000000, fc1=256, fc2=256, batch_size=100, reward_scale=2): 
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_size, input_dims, self.n_actions)
        
        self.actor = Actor(max_action=env.action_space.high, alpha=alpha, input_dims=input_dims,
                            fc1=fc1, fc2=fc2, n_actions=n_actions, name = env_id+'_actor')
        self.critic1 = Critic(beta, input_dims, fc1, fc2, n_actions, name=env_id+'_critic1')
        self.critic2 = Critic(beta, input_dims, fc1, fc2, n_actions, name=env_id+'_critic2')
        self.value = Value_Network(input_dims, fc1, fc2, beta, name=env_id+'value')
        self.target_value = Value_Network(input_dims, fc1, fc2, beta, name=env_id+'_tarvet_val')

        self.scale= reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state=T.tensor([observation]).to(self.actor.device)
        action, _ = self.actor.sample_normal(state, reparamaterize=False)
        return action.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau=self.tau
        
        value_params = self.value.named_parameters()
        target_value_params = self.target_value.named_parameters()

        value_param_dict = dict(value_params)
        target_value_param_dict=dict(target_value_params)

        for name in value_param_dict:
            value_param_dict[name] = tau * value_param_dict[name].clone() + \
                (1-tau) * target_value_param_dict[name].clone()
        self.value.load_state_dict(value_param_dict)

    def save_models(self):
        self.actor.save()
        self.value.save()
        self.target_value.save()
        self.critic1.save()
        self.critic2.save()
    
    def load_models(self):
        self.actor.load()
        self.value.load()
        self.target_value.load()
        self.critic1.load()
        self.critic2.load()

    def learn(self):
        if self.memory.mem_count < self.batch_size:
            return
        state, action, reward, state_, done = \
            self.memory.sample_buffer(self.batch_size)
        state= T.tensor(state, dtype=T.float).to(self.critic1.device)
        action=T.tensor(action, dtype=T.float).to(self.critic1.device)
        reward=T.tensor(reward, dtype=T.float).to(self.critic1.device)
        state_=T.tensor(state_, dtype=T.float).to(self.critic1.device)
        done = T.tensor(done).to(self.critic1.device)

        value = self.value(state).view(-1)
        value_= self.target_value(state_).view(-1)
        value_[done]=0.0

        #get the value loss
        actions, log_probs = self.actor.sample_normal(state, reparamaterize=False)
        log_probs=log_probs.view(-1)
        q1 = self.critic1.forward(state, actions)
        q2 = self.critic2.forward(state, actions)
        qval = T.min(q1, q2)
        qval=qval.view(-1)

        self.value.optimizer.zero_grad()
        val_target = qval - log_probs
        val_loss = .5*F.mse_loss(value, val_target)
        val_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        #get actor loss
        actions, log_probs= self.actor.sample_normal(state, reparamaterize=False)
        log_probs=log_probs.view(-1)
        new_policy_q1=self.critic1.forward(state, actions)
        new_policy_q2=self.critic1.forward(state, actions)
        q_new = T.min(new_policy_q1, new_policy_q2)
        q_new=q_new.view(-1)

        self.actor.optimizer.zero_grad()
        act_target = log_probs - q_new
        act_target = T.mean(act_target)
        act_target.backward(retain_graph=True)
        self.actor.optimizer.step()

        #critic loss
        q_hat = self.scale * reward + self.gamma * value_

        q1_old = self.critic1.forward(state, action).view(-1)
        q2_old = self.critic1.forward(state, action).view(-1)
        q1_loss = .5 * F.mse_loss(q1_old, q_hat)
        q2_loss = .5 * F.mse_loss(q2_old, q_hat)

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss = q1_loss + q2_loss
        critic_loss.backward(retain_graph=True)
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_network_parameters()









