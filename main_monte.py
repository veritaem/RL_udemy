import gym
from monte_carlo import Agent

if __name__=='__main__':
    env=gym.make("Blackjack-v0")
    agent=Agent()
    n_episodes=500000
    for i in range(n_episodes):
        if i%50000==0:
            print(f'starting episode {i}')
        observation=env.reset()
        done=False
        while not done:
            #choose an action based on the policy
            action=agent.policy(observation)
            # take the action
            observation_, reward, done, info= env.step(action)
            agent.memory.append((observation, reward))
            observation=observation_
        agent.update_V()
    print(agent.V[(21, 3, True)])
    print(agent.V[(4, 1, False)])

