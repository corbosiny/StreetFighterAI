import gym
env = gym.make('LunarLander-v2')
env.reset()

OBSERVATION_INDEX = 0
REWARD_INDEX = 1
DONE_INDEX = 2
INFO_INDEX = 3

for _ in range(30):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    if done == True:
        break
 
env.close()
