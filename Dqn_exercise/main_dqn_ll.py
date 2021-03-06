from lunarlandermodel import Agent
import numpy as np
import gym
import tensorflow as tf

if __name__=='__main__':
    tf.compat.v1.disable_eager_execution()
    env=gym.make('LunarLander-v2')
    lr=0.001
    n_Games=500
    agent=Agent(gamma=0.99, epsilon=1.0, lr=lr, 
                input_dims=env.observation_space.shape[0],
                n_actions = env.action_space.n, mem_size=1000000, batch_size=64,
                epsilon_end=0.01)
    scores=[]
    eps_history=[]
    for i in range(n_Games):
        done=False
        score=0
        observation=env.reset()
        while not done:
            action=agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score+=reward
            agent.store_transition(observation,reward,action, observation_, done)
            observation=observation_
            agent.learn()
            eps_history.append(agent.epsilon)
            scores.append(score)

            avg_scores=np.mean(scores[-100:])
            print(f'episode:{i}, score:{score}, avg scoring:{avg_scores}, epsilon-{agent.epsilon}')

            filename= 'lunarlander_tf2.png'
            


