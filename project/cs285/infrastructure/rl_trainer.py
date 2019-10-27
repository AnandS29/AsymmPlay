import time

from collections import OrderedDict
import pickle
import numpy as np
import tensorflow as tf
import gym
import os

from cs285.infrastructure.utils import *
from cs285.infrastructure.tf_utils import create_tf_session
from cs285.infrastructure.logger import Logger

import random
import numpy as np
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
from gym_minigrid.wrappers import *

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below

class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger, create TF session
        self.params = params
        self.logger = Logger(self.params['logdir'])
        self.sess = create_tf_session(self.params['use_gpu'], which_gpu=self.params['which_gpu'])

        # Set random seeds
        seed = self.params['seed']
        tf.set_random_seed(seed)
        np.random.seed(seed)

        #############
        ## ENV
        #############

        # Make the gym environment
        #self.env = FlatObsWrapper(gym.make(self.params['env_name']))
        #self.env = gym.make(self.params['env_name'])
        #self.env.seed(seed)

        eval_seed = 10

        self.env_teacher, self.env_student = gym.make("MiniGrid-TeacherEnv-5x5-v0"), gym.make("MiniGrid-StudentEnv-5x5-v0")

        self.eval_env = gym.make("MiniGrid-StudentEnv-5x5-v0")
        # env_student = gym.make("MiniGrid-StudentEnv-5x5-v0")
        self.eval_env.goal_pos = [5,5]
        # # env_student.teacher_step_count = teacher_step_count
        self.eval_env.setup()
        self.eval_env.seed(eval_seed)

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env_teacher.spec.max_episode_steps
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env_teacher.action_space, gym.spaces.Discrete)
        #print(self.env.action_space)
        #print("DIS",spaces.Discrete(6))
        #print("HIIII")
        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes
        #print(self.env.observation_space)
        print(self.env_teacher.observation_space)
        ob_dim = self.env_teacher.observation_space.shape[0]
        ac_dim = self.env_teacher.action_space.n if discrete else self.env_teacher.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        # if 'model' in dir(self.env):
        #     self.fps = 1/self.env.model.opt.timestep
        # else:
        #     self.fps = self.env.env.metadata['video.frames_per_second']

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent_teacher = agent_class(self.sess, self.env_teacher, self.params['agent_params'])

        agent_class = self.params['agent_class']
        self.agent_student = agent_class(self.sess, self.env_student, self.params['agent_params'])

        #############
        ## INIT VARS
        #############

        tf.global_variables_initializer().run(session=self.sess)

    def run_training_loop(self, n_iter, collect_policy_teacher, collect_policy_student, eval_policy,
                        initial_expertdata=None, relabel_with_expert=False,
                        start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(itr,
                                initial_expertdata, collect_policy_teacher, collect_policy_student,
                                self.params['batch_size'])
            paths_teacher, paths_student, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            self.agent_teacher.add_to_replay_buffer(paths_teacher)
            self.agent_student.add_to_replay_buffer(paths_student)

            # train agent (using sampled data from replay buffer)
            self.train_agent()

            # log/save
            if (self.log_video or self.log_metrics): # Added False

                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths_teacher, paths_student, collect_policy_teacher, collect_policy_student, train_video_paths)


                if self.params['save_params']:
                    # save policy
                    print('\nSaving agent\'s actor...')
                    self.agent.actor.save(self.params['logdir'] + '/policy_itr_'+str(itr))

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, load_initial_expertdata, collect_policy_teacher, collect_policy_student, batch_size):
        # TODO: GETTHIS from HW1
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # TODO decide whether to load training data or use
        # HINT: depending on if it's the first iteration or not,
            # decide whether to either
                # load the data. In this case you can directly return as follows
                # ``` return loaded_paths, 0, None ```

                # collect data, batch_size is the number of transitions you want to collect.
        # TODO collect data to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']

        if itr == 0:
            if load_initial_expertdata is not None:
                paths = pickle.load(open(self.params['expert_data'], 'rb'))
                return paths, 0, None

        print("\nCollecting data to be used for training...")
        paths_teacher, paths_student, envsteps_this_batch = sample_trajectories(self.env_teacher, self.env_student, collect_policy_teacher, collect_policy_student, batch_size, self.params['ep_len'])
        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths_teacher, paths_student, envsteps_this_batch, train_video_paths

    def train_agent(self):
        # TODO: GETTHIS from HW1
        print('\nTraining agent using sampled data from replay buffer...')
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            for ag in [self.agent_teacher, self.agent_student]:
                # TODO sample some data from the data buffer
                # HINT1: use the agent's sample function
                # HINT2: how much data = self.params['train_batch_size']
                ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = ag.sample(self.params['train_batch_size'])

                # TODO use the sampled data for training
                # HINT: use the agent's train function
                # HINT: print or plot the loss for debugging!
                for _ in range(self.params['num_grad_steps']):
                    ag.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
                #loss = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
                #self.latest_loss = loss
                #print("Loss:", self.sess.run(self.agent.actor.loss))

    def do_relabel_with_expert(self, expert_policy, paths):
        # TODO: GETTHIS from HW1 (although you don't actually need it for this homework)
        for i in range(len(paths)):
            acs = expert_policy.get_action(paths[i]["observation"])
            paths[i]["action"] = acs
        return paths

    ####################################
    ####################################

    def perform_logging(self, itr, paths_teacher, paths_student, collect_policy_teacher, collect_policy_student, train_video_paths):

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths_student, eval_envsteps_this_batch = sample_trajectories_eval(self.eval_env, collect_policy_student, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            eval_paths = eval_paths_student
            train_returns_student = [path["reward"].sum() for path in paths_student]
            train_returns_teacher = [path["reward"].sum() for path in paths_teacher]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens_student = [len(path["reward"]) for path in paths_student]
            train_ep_lens_teacher = [len(path["reward"]) for path in paths_teacher]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn_Student"] = np.mean(train_returns_student)
            logs["Train_StdReturn_Student"] = np.std(train_returns_student)
            logs["Train_MaxReturn_Student"] = np.max(train_returns_student)
            logs["Train_MinReturn_Student"] = np.min(train_returns_student)
            logs["Train_AverageEpLen_Student"] = np.mean(train_ep_lens_student)

            logs["Train_AverageReturn_Teacher"] = np.mean(train_returns_teacher)
            logs["Train_StdReturn_Teacher"] = np.std(train_returns_teacher)
            logs["Train_MaxReturn_Teacher"] = np.max(train_returns_teacher)
            logs["Train_MinReturn_Teacher"] = np.min(train_returns_teacher)
            logs["Train_AverageEpLen_Teacher"] = np.mean(train_ep_lens_teacher)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time


            if itr == 0:
                self.initial_return_student = np.mean(train_returns_student)
                self.initial_return_teacher = np.mean(train_returns_teacher)
            logs["Initial_DataCollection_AverageReturn_Student"] = self.initial_return_student
            logs["Initial_DataCollection_AverageReturn_Teacher"] = self.initial_return_teacher

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
