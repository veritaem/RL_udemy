import gym
import numpy as np
import matplotlib.pyplot as plt 
from monte_control import Agent

if __name__=='__main__':
    env=gym.make('Blackjack-v0')
    agent=Agent(epsilon=.001)
    episodes=200000
    win_lose_draw={-1:0, 0:0, 1:0}
    win_rates=[]
    for i in range(episodes):
        if i > 0 and i % 1000==0:
            pct=win_lose_draw[1]/i
            win_rates.append(pct)
        if i%50000==0:
            rates=win_rates[-1] if win_rates else 0.0
            print(f'starting episode {i}, win rate {rates}')
        observation=env.reset()
        done=False
        while not done:
            action=agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.memory.append((observation, action, reward))
            observation=observation_
            agent.update_Q()
            win_lose_draw[reward]+=1
    plt.plot(win_rates)
    plt.show()

