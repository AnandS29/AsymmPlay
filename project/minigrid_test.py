import random
import numpy as np
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX

# Test specifically importing a specific environment
from gym_minigrid.envs import DoorKeyEnv

# Test importing wrappers
from gym_minigrid.wrappers import *
env_name = env_list[0]
#print(env_list)
env = gym.make("MiniGrid-DoorKey-16x16-v0")
#print(env.observation_space.spaces["image"].shape)
#obs_space.spaces["image"].shape
for i_episode in range(10):
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Done")
            break
env.close()
