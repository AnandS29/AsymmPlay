import random
import numpy as np
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
import time

# Test specifically importing a specific environment
from gym_minigrid.envs import DoorKeyEnv

# Test importing wrappers
from gym_minigrid.wrappers import *
# env_name = env_list[0]
# #print(env_list)
# env = gym.make("MiniGrid-DoorKey-16x16-v0")
# #print(env.observation_space.spaces["image"].shape)
# #obs_space.spaces["image"].shape
# for i_episode in range(10):
#     observation = env.reset()
#     for t in range(1000):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             print("Done")
#             break
# env.close()

seed=10

env_teacher = gym.make("MiniGrid-TeacherEnv-5x5-v0")
env_teacher.seed(seed)
ob = env_teacher.reset() # HINT: should be the output of resetting the env
max_path_length = 100
env_teacher.max_steps = max_path_length

# init vars
obs_t, acs_t, rewards_t, next_obs_t, terminals_t, image_obs_t = [], [], [], [], [], []
obs_s, acs_s, rewards_s, next_obs_s, terminals_s, image_obs_s = [], [], [], [], [], []
steps = 0
# start, end, teacher_step_count = None, None, None
# gamma = 1
# print("Teacher")
# print(env_teacher.action_space)
# while True:
#     obs_t.append(ob)
#     env_teacher.render()
#     ac = env_teacher.action_space.sample() # HINT: query the policy's get_action function
#     acs_t.append(ac)
#     ob, rew, done, info = env_teacher.step(ac)
#     print(done)
#     steps += 1
#     next_obs_t.append(ob)
#     rewards_t.append(rew)
#     rollout_done = (1 if (steps>=max_path_length or done) else 0) # HINT: this is either 0 or 1
#     terminals_t.append(rollout_done)
#
#     if rollout_done and steps>=200:
#         start, end, teacher_step_count = info["start"], info["end"], info["teacher_step_count"]
#         break
# env_teacher.close()
for i in range(4):
    env_student = gym.make("MiniGrid-StudentEnv-5x5-v0")
    # env_student = gym.make("MiniGrid-StudentEnv-5x5-v0")
    env_student.goal_pos = [3,1]
    # # env_student.teacher_step_count = teacher_step_count
    env_student.setup()
    env_student.seed(seed)
    ob = env_student.reset() # HINT: should be the output of resetting the env

    # init vars
    steps = 0
    print("Student")
    while True:
        obs_t.append(ob)
        env_student.render()
        ac = env_student.action_space.sample() # HINT: query the policy's get_action function
        print(ac,end=",")
        acs_t.append(ac)

        # take that action and record results
        ob, rew, done, info = env_student.step(ac)

        # record result of taking that action
        steps += 1
        next_obs_s.append(ob)
        rewards_s.append(rew)

        # TODO end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = (1 if (steps>=max_path_length or done) else 0) # HINT: this is either 0 or 1
        terminals_s.append(rollout_done)

        if rollout_done:
            break

    env_student.close()
rewards_s = [-1*gamma*float(r) for r in rewards_s]
teacher_r = [0]*len(rewards_t)
teacher_r[-1] = gamma * max([0, t_b-t_a])
