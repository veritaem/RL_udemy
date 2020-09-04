import gym
import matplotlib.pyplot as plt 
import numpy as np
from actor_critic import Agent, Actor_critic

def plot_learning(scores, x, fig_file):
    running_avg = np.zeros(len(scores))
    for i in running_avg:
        running_avg[i] = np.mean(scores[max(0, i-100):i+1])
    plt.plot(x, running_avg)
    plt.title('running average-100 games')
    plt.savefig(fig_file)

if __name__=='__main__':
    env=gym.make('LunarLander-v2')
    n_games=2000
    lr=0.00005
    fc1=2048
    fc2=1536
    gamma=0.99
    fname=f'plots/lunarlander_actor_critic_{fc1}_{fc2}_lr{lr}_{n_games}games.png'
    scores=[]
    agent = Agent(4, [8], lr=lr, fc1=fc1, fc2=fc2, gamma=gamma)
    for i in range(n_games):
        observation=env.reset()
        done=False
        score=0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score+=reward
            agent.learn(observation, reward, observation_, done)
            observation=observation_
        scores.append(score)
        avg= np.mean(scores[-100:])
        print(f'episode:{i}, score:{score}, average:{avg}')
    x= [(i+1) for i in len(range(scores))]
    plot_learning(x, scores, fname)





