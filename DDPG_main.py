import gym
import numpy as np
import matplotlib.pyplot as plt 
from DDPG_Agent import Agent

def plot_learning(x, scores, fig_file):
    running_avg = np.zeros(len(scores))
    for i in running_avg:
        running_avg[i]=np.mean(scores[max(0, i-100):i+1])
    plt.plot(scores, x)
    plt.title("Running avg-100 games")
    plt.savefig(fig_file)

if __name__=='__main__':
    env=gym.make('LunarLanderContinuous-v2')
    #hyperparams
    lralpha=.0001
    lrbeta=.001
    gamma=0.99
    tau=1
    max_size=1000000
    fc1d=400
    fc2d=300
    batch=64

    n_games=1000
    agent=Agent(tau=tau, alpha=lralpha, beta = lrbeta, input_dims=env.observation_space.shape,
                n_actions=4, gamma=gamma, max_size=max_size, fc1_dims=fc1d,
                fc2_dims=fc2d, batch_size=batch)
    best_score = env.reward_range[0]
    scores=[]
    for i in range(n_games):
        observation=env.reset()
        done=False
        agent.noise.reset()
        score=0
        while not done:
            action=agent.choose_action(observation)
            observation_, reward, done, info= env.step(action)
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            score+=reward
            observation=observation_
        scores.append(score)
        avg=np.mean(scores[-100:])
        if avg > best_score:
            best_score = avg
            agent.save_models()
            print(f'new high score! {best score}')
        print(f'episode:{i}, score:{score}, average:{avg}')
    x = [(i+1) for i in range(n_games)]
    fig_file = f'plots/DDPG_lunarlander_{lralpha}_{lrbeta}_{tau}_{n_games}_games.png'
    plot_learning(x, scores, fig_file)

