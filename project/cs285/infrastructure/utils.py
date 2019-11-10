import numpy as np
import time
import scipy

############################################
############################################

def sample_trajectory(env_teacher, env_student, policy_teacher, policy_student, max_path_length, render=False, render_mode=('rgb_array'), gamma=1.0, seed=10):

    # initialize env for the beginning of a new rollout
    env_teacher.seed(seed)
    ob = env_teacher.reset() # HINT: should be the output of resetting the env
    env_student.seed(seed)
    env_teacher.max_steps = max_path_length
    env_student.max_steps = max_path_length

    # init vars
    obs_t, acs_t, rewards_t, next_obs_t, terminals_t, image_obs_t = [], [], [], [], [], []
    obs_s, acs_s, rewards_s, next_obs_s, terminals_s, image_obs_s = [], [], [], [], [], []
    steps = 0
    start, end, teacher_step_count = None, None, None
    while True:
        obs_t.append(ob)
        ac = policy_teacher.get_action(ob) # HINT: query the policy's get_action function
        ac = ac[0]
        acs_t.append(ac)

        # take that action and record results
        ob, rew, done, info = env_teacher.step(ac)

        # record result of taking that action
        steps += 1
        next_obs_t.append(ob)
        rewards_t.append(rew)

        # TODO end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = (1 if (steps>=max_path_length or done) else 0) # HINT: this is either 0 or 1
        terminals_t.append(rollout_done)

        if rollout_done:
            end = info["agent_pos"]
            teacher_step = steps
            break

    env_student.end_pos = end
    env_student.teacher_step = teacher_step
    env_student.seed(seed)
    ob = env_student.reset() # HINT: should be the output of resetting the env
    # init vars
    steps = 0
    while True:
        # use the most recent ob to decide what to do
        obs_s.append(ob)
        ac = policy_student.get_action(ob) # HINT: query the policy's get_action function
        ac = ac[0]
        acs_s.append(ac)

        ob, rew, done, info = env_student.step(ac)

        steps += 1
        next_obs_s.append(ob)
        rewards_s.append(rew)

        rollout_done = (1 if (steps>=max_path_length or done) else 0) # HINT: this is either 0 or 1
        terminals_s.append(rollout_done)

        if rollout_done:
            break

    teacher_r = [0]*len(rewards_t)
    teacher_r[-1] = max([0, rewards_s[-1] - rewards_t[-1]])

    return Path(obs_t, image_obs_t, acs_t, teacher_r, next_obs_t, terminals_t), Path(obs_s, image_obs_s, acs_s, rewards_s, next_obs_s, terminals_s)

def sample_trajectories(env_teacher, env_student, policy_teacher, policy_student, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):

    # TODO: GETTHIS from HW1
    """s
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths_t, paths_s = [], []
    while timesteps_this_batch < min_timesteps_per_batch:
        path_teacher, path_student = sample_trajectory(env_teacher, env_student, policy_teacher, policy_student, max_path_length, render, render_mode)
        paths_t.append(path_teacher)
        paths_s.append(path_student)
        timesteps_this_batch += get_pathlength(path_teacher) + get_pathlength(path_student)

    #print("OBS")
    #print([p["observation"].shape for p in paths_s])
    return paths_t, paths_s, timesteps_this_batch

def sample_trajectories_eval(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    # TODO: GETTHIS from HW1
    """s
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory_eval(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch

def sample_trajectory_eval(env, policy, max_path_length, render=False, render_mode=('rgb_array')):

    # initialize env for the beginning of a new rollout
    ob = env.reset() # TODO: GETTHIS from HW1
    env.seed(10)
    env.teacher_step_count = 0
    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    p = np.random.random()
    while True:

        # render image of the simulated env
        if p <= 0.2 and False:
            env.render()
            time.sleep(0.05)
        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob) # TODO: GETTHIS from HW1
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # End the rollout if the rollout ended
        # Note that the rollout can end due to done, or due to max_path_length
        rollout_done = (1 if (steps>max_path_length or done) else 0) # TODO: GETTHIS from HW1
        terminals.append(rollout_done)

        if rollout_done:
            break
    gamma = 1
    rewards = [-1*gamma*float(r) for r in rewards]
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):

    # TODO: GETTHIS from HW1
    """
        Collect ntraj rollouts.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    paths = []

    for _ in range(ntraj):
        #paths.append(sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render, render_mode)[0])
        paths.append(sample_trajectory(env, policy, max_path_length, render, render_mode)[1])

    return paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    #print([p["observation"] for p in paths[:2]])
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    #print("DONE")
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])
