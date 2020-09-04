import numpy as np
from DDPGHelpers import OUActionNoise, ReplayBuffer
from DDPG_actor_critic import Actor, Critic
import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.099, 
                max_size=1000000, fc1_dims=400, fc2_dims=300, batch_size=64):
        self.alpha = alpha
        self.beta = beta
        self.gamma=gamma
        self.input_dims= input_dims
        self.tau= tau
        self.replay_memory=ReplayBuffer(max_size, input_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.actor =Actor(input_shape=self.input_dims, alpha= alpha, name='actor',
                        fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions)
        self.critic = Critic(beta=beta, input_dims=input_dims, fc1=fc1_dims, fc2=fc2_dims,
                            n_actions=n_actions, name='t_critic')
        self.t_actor =Actor(input_shape=self.input_dims, alpha= alpha, name='actor',
                        fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=n_actions)
        self.t_critic = Critic(beta=beta, input_dims=input_dims, fc1=fc1_dims, fc2=fc2_dims,
                            n_actions=n_actions, name='t_critic')
        self.update_network_params(tau=1)
    
    def choose_action(self, observation):
        self.actor.eval()
        state = T.Tensor([observation], dtype=T.float).to(self.device)
        mu=self.actor.forward(state).to(self.device)
        mu_prime = mu+T.Tensor(self.noise(), dtype=np.float).to(self.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, state_, done):
        self.replay_memory.store_transition(state, action, reward, state_, done)
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.t_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.t_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.t_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.t_critic.load_checkpoint()

    def learn(self):
        if self.replay_memory.mem_count < batch_size:
            return
        states, actions, rewards, dones = \
            self.replay_memory.sample_buffer(self.batch_size)
        states =T.Tensor(states, dtype=T.float).to(self.agent.device)
        actions=T.Tensor(actions, dtype=T.float).to(self.agent.device)
        states_=T.Tensor(states_, dtype=T.float).to(self.agent.device)
        rewards= T.Tensor(rewards, dtype=T.float).to(self.agent.device)
        dones=T.Tensor(dones).to(self.device)

        critic_values = self.critic.forward(states, actions)
        target_actions = self.t_actor.forward(states_)
        critic_values_ = self.t_critic.forward(states_, target_actions)

        critic_values_[dones]=0.0
        critic_values_=critic_values_.view(-1)

        target = reward + gamma * critic_values_
        target=target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        closs = F.mse_loss(target, critic_values)
        closs.backward()
        self.critic.optimizer.step()
        self.actor.optimizer.zero_grad()
        aloss = -self.critic.forward(states, self.actor.forward(states))
        aloss=T.mean(aloss)
        aloss.backward()
        self.actor.optimizer.step()

        self.update_network_params()

    def update_network_params(self, tau=None):
        #update for actor and critic<- tau *current weights + (1-tau)*old weight
        if tau is None:
            tau = self.tau
        actor_params = self.actor.named_parameters()
        critic_params= self.critic.named_parameters()
        target_actor_params = self.t_actor.named_parameters()
        target_critic_params=self.t_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict=dict(critic_params)
        tactor_state_dict=dict(target_actor_params)
        tcritic_state_dict=dict(target_critic_params)
        for name in critic_state_dict:
            critic_state_dict[name]=tau*self.critic_state_dict[name].clone() +\
                (1-tau) * self.tcritic_state_dict[name].clone()
        for name in actor_state_dict:
            actor_state_dict[name]=tau * self.actor_state_dict[name].clone() +\
                (1-tau) * self.tactor_state_dict[name].clone()

        self.t_actor.load_state_dict(actor_state_dict)
        self.t_critic.load_state_dict(critic_state_dict)

        

    