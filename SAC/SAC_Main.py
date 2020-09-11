import numpy as np
import matplotlib.pyplot as plt 
import gym
import pybullet_envs
from SAC_Agent import Agent

def plot_learning(x, scores, fig_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i]=np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title("Running avg-100 games")
    plt.savefig(fig_file)


if __name__=='__main__':
    env_id = 'InvertedPendulumBulletEnv-v0'
    env = gym.make(env_id)
    alpha, beta, gamma, tau = 3e-4, 3e-4, 0.99, 5e-3
    n_actions = env.action_space.shape[0]
    input_dims = env.observation_space.shape
    reward_scale = 2
    batch_size, fc1, fc2 = 256, 256, 256
    max_size=1000000

    agent = Agent(alpha, beta, tau, input_dims, n_actions, env, env_id, 
                gamma, max_size, fc1, fc2, batch_size, reward_scale)
    n_games =250
    filename = f'{env_id}_{n_games}_scale_{agent.scale}.png'
    figfile = f'plots/{filename}'
    best_score = env.reward_range[0]
    load_checkpoint=False
    scores = []

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
    else:
        env.render(mode='human')
    steps=0
    for i in range(n_games):
        observation = env.reset()
        done =False
        score=0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info=env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            steps+=1
            score+=reward
            if not load_checkpoint:
                agent.learn()
        scores.append(score)
        avg = np.mean(scores[-100:])
        if avg > best_score:
            best_score = avg
            if not load_checkpoint:
                agent.save_models()
        print(f'episode:{i}, score:{score}, running avg:{avg}, steps:{steps}')
    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning(x, scores, figfile)





