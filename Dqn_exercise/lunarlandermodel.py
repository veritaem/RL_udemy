import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.optimizers import Adam
from keras.models import load_model
import os

class ReplayBuffer():
    def __init__(self, mem_size, input_dims):
        self.mem_size=mem_size
        self.mem_cntr=0
        self.state_memory=np.zeros((self.mem_size, *input_dims),
                                    dtype=np.float32)
        self.new_state_memory=np.zeros((self.mem_size, *input_dims), 
                                    dtype=np.float32)
        self.action_memory=np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory=np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory=np.zeros(self.mem_size, dtype=np.int32)
    
    def store_transition(self, state, reward, action, state_, done):
        index=self.mem_cntr % self.mem_size
        self.state_memory[index]=state
        self.terminal_memory[index]=1-int(done)
        self.reward_memory[index]=reward
        self.action_memory[index]=action
        self.new_state_memory[index]=state_
        self.mem_cntr+=1

    def sample_buffer(self, batch_size):
        max_mem=min(self.mem_cntr, self.mem_size)
        batch=np.random.choice(max_mem, batch_size, replace=False)
        states=self.state_memory[batch]
        states_=self.new_state_memory[batch]
        rewards=self.reward_memory[batch]
        actions=self.action_memory[batch]
        terminal=self.terminal_memory[batch]

        return states, rewards, actions, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model=keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None )
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model

class Agent(object):
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3,
                epsilon_end=0.01, mem_size=1000000, fname='dqn_model.hb'):
        self.action_space=[i for i in range(n_actions)]
        self.gamma=gamma
        self.epsilon=epsilon
        self.eps_end=epsilon_end
        self.eps_dec=epsilon_dec
        self.batch_size=batch_size
        self.model_file=fname
        self.memory=ReplayBuffer(mem_size, input_dims)
        self.q_eval=build_dqn(lr, n_actions, input_dims, 256, 256)
    
    def store_transition(self, state, reward, action, new_state, done):
        self.memory.store_transition(state, reward, action, new_state, done)
    
    def choose_action(self, observation):
        if np.random.random()<self.epsilon:
            action=np.random.choice(self.action_space)
        else:
            state=np.array([observation])
            actions=self.q_eval.predict(state)
            action= np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        states, rewards, actions, states_, dones = \
            self.memory.sample_buffer(self.batch_size)
        q_eval= self.q_eval.predict(states)
        q_next= self.q_eval.predict(states_)

        q_target=np.copy(q_eval)
        batch_index=np.arange(self.batch_size, dtype=float32)

        q_target[batch_index, actions] = rewards + \
            self.gamma*np.max(q_next, axis=1)*dones

        self.q_eval.train_on_batch(states, q_target)
        self.epsilon= self.epsilon-self.eps_dec if self.epsilon>\
            self.eps_end else self.eps_end

        def save_model(self):
            self.q_eval.save(self.model_file)

        def load_model(self):
            self.q_eval=load_model(self.model_file)





