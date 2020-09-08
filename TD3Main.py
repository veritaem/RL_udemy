import gym
import matplotlib.pyplot as plt 
import numpy as np
from TD3Agent import Agent

def plot_learning(x, scores, fig_file):
    running_avg = np.zeros(len(scores))
    for i in running_avg:
        running_avg[i]=np.mean(scores[max(0, i-100):i+1])
    plt.plot(scores, x)
    plt.title("Running avg-100 games")
    plt.savefig(fig_file)

if __name__=='__main__':
    env = gym.make('BipedalWalker-v3')
    n_games =1500
    alpha, beta, tau, gamma = 1e-3, 1e-3, 1, 0.99
    agent = Agent(input_dims=env.observation_space.shape, fname= 'tmp/agent', name = 'agent',
                    alpha=alpha, beta=beta, tau=tau, gamma=gamma, env=env)
    
    best_score = env.reward_range[0]
    scores = []
    for i  in range(n_games):
        observation=env.reset()
        env.render()
        done=False
        score=0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score+=reward
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

        scores.append(score)
        avg=np.mean(scores[-100:])
        if avg > best_score:
            best_score = avg
            agent.save_models()
            print(f'new high score! {best_score}')
        print(f'episode:{i}, score:{score}, average:{avg}')
    x = [(i+1) for i in range(n_games)]
    fig_file = f'plots/TD3Biped_{alpha}_{beta}_{tau}_{n_games}_games.png'
    plot_learning(x, scores, fig_file)
