import gym
env = gym.make('CartPole-v0')
env.reset()

OBSERVATION_INDEX = 0
REWARD_INDEX = 1
DONE_INDEX = 2
INFO_INDEX = 3

for _ in range(30):
    env.render()
    result = env.step(env.action_space.sample())
    if result[DONE_INDEX] == True:
        break
 
env.close()
