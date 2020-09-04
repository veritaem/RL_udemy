import numpy as np

class Agent():
    def __init__(self, lr, gamma, n_actions, state_space,
                e_s, e_e, e_d):
        self.gamma=gamma
        self.lr=lr
        self.e_s=e_s
        self.e_e=e_e
        self.e_d=e_d
        self.state_space=state_space
        self.action_space=[i for i in range(n_actions)]
        self.Q={}
        self.init_Q()

    def init_Q(self): # for each action possible in each state, start with nothing
        for state in self.state_space:
            for action in self.action_space:
                self.Q[(state, action)]=0.0
    
    def max_action(self, state): # for all actions in this state, get the best weve seen 
        actions=np.array([self.Q[(state, a)] for a in self.action_space])
        max_action=np.argmax(actions)
        return max_action
    
    def choose_action(self, state): # pick a random action or the best action we know of in the state
        if np.random.random() < self.e_s:
            action=np.random.choice(self.action_space)
        else:
            action = self.max_action(state)
        return action

    def decrement_epsilon(self): # epsilon - its decrement until it hits the minimum
        self.e_s=self.e_s-self.e_d if self.e_s>self.e_e else self.e_e

    def learn(self, state, action, reward, state_): #update agent valuation of the move by 
        #the moves old values, the current reward, and the discounted future returns 
        a_max = self.max_action(state_)
        self.Q[(state, action)]= self.Q[(state, action)] + self.lr * \
                                        (reward + self.gamma * \
                                self.Q[(state_, a_max)]-self.Q[(state, action)])
    
