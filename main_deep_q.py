import gym
import numpy as np
import matplotlib.pyplot as plt
from deep_Q_cart import Agent

class CartPoleDigitizer:
    def __init__(self, bounds=(2.4, 4, 0.209, 4), n_bins=10):
        self.position_space = np.linspace(-1*bounds[0], bounds[0], n_bins)
        self.velocity_space = np.linspace(-1*bounds[1], bounds[1], n_bins)
        self.pole_angle_space = np.linspace(-1*bounds[2], bounds[2], n_bins)
        self.pole_velocity_space = np.linspace(-1*bounds[3], bounds[3], n_bins)
        self.states=self.get_state_space()

    def get_state_space(self):
        states=[]
        for i in range(len(self.position_space)+1):
            for j in range(len(self.velocity_space)+1):
                for k in range(len(self.pole_angle_space)+1):
                    for l in range(len(self.pole_velocity_space)+1):
                        states.append((i, j, k , l))
        return states

    def digitize(self, observation):
        x, x_dot, theta, theta_dot= observation
        cart_x= int(np.digitize(x, self.position_space))
        cart_x_dot=int(np.digitize(x_dot, self.velocity_space))
        pole_theta = int(np.digitize(theta, self.pole_angle_space))
        pole_theta_dot = int(np.digitize(theta_dot, self.pole_velocity_space))

        return (cart_x, cart_x_dot, pole_theta, pole_theta_dot)

def plot_learning(scores,x):
    running_avg=np.zeros(len(scores))
    for i in range(len(scores)):
        running_avg[i]=np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running avg(100 game history)')
    plt.show()

if __name__=='__main__':
    env=gym.make('CartPole-v0')
    n_games=50000
    eps_dec=2/n_games
    digitizer=CartPoleDigitizer()
    agent=Agent(lr=0.01, gamma=0.99, n_actions=2, e_s=1.0, e_e=0.01, e_d=eps_dec,
                state_space=digitizer.states)
    scores=[]
    for i in range(n_games):
        observation=env.reset()
        done=False
        score=0
        state=digitizer.digitize(observation)
        while not done:
            action=agent.choose_action(state)
            observation_,reward, done, info=env.step(action)
            state_=digitizer.digitize(observation_)
            agent.learn(state, action, reward, state_)
            state=state_
            score+=reward
        if i%5000==0:
            print(f'episode {i}, score {score}, epsilon {agent.e_s}')
        agent.decrement_epsilon()
        scores.append(score)
    x=[(i+1) for i in range(n_games)]
    plot_learning(scores, x)
    
