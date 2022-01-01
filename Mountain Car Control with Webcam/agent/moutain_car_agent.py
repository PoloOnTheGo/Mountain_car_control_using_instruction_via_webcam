from gym.utils.play import play
import pygame
from agent import agent_prototype
import numpy as np

# keys_to_action = {ord('t'): 2, ord('z'): 0, ord('o'): 1}

global env_name
env_name = 'MountainCar-v0'
list = np.zeros(150)
env = agent_prototype.initialize_environment('MountainCar-v0')
play(env)
env.close()

# tot_reward = 0
# for i_episode in range(150):
#     state = env.reset()
#     for t in range(1000):
#         env.render()
#         print(state)
#         # action = int(list[i_episode])
#         # if t%10 ==0:
#         #     action +=2
#         action = int(list[i_episode])
#         print(action)
#         state, reward, done, info = env.step(action)
#         tot_reward += reward
#         if done or state[0] >= 0.5:
#             print('Episode {} Average Reward: {}'.format(i_episode + 1, reward))
#             break
# env.close()
