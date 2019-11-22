from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX
import sys
import collections
import numpy


class DoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

class TeacherDoorKeyEnv(DoorKeyEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        self.end_pos = None
        self.is_teaching = True
        super().__init__(
            size=size
        )
        self.min_steps = 0

    def step(self, action):

        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        #print(action)
        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal' and not self.is_teaching:
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            if self.step_count >= self.min_steps and self.is_teaching:
                done = True
            else:
                pass
        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()
        if done and self.is_teaching:
            student_return_avg = []
            envs = []
            for i in range(self.args.procs):
                env = gym.make(self.args.env)
                env.seed(self.args.seed)
                env.is_teaching = False
                env.end_pos = self.agent_pos
                envs.append(env)
            update = 0
            num_frames = 0

            md_index = np.random.choice(range(len(self.student_hist_models)),1)[0]
            if self.args.historical_averaging:
                md = copy.deepcopy(self.student_hist_models[md_index])
            else:
                md = self.student_hist_models[md_index]

            # while num_frames < self.args.frames and update<5:
            while update < args.s_iters_per_teaching:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                algo = torch_ac.PPOAlgo(envs, md, device, self.args.frames_per_proc, self.args.discount, self.args.lr, self.args.gae_lambda,
                                        self.args.entropy_coef, self.args.value_loss_coef, self.args.max_grad_norm, self.args.recurrence,
                                        self.args.optim_eps, self.args.clip_eps, self.args.epochs, self.args.batch_size, self.preprocess_obss)
                update_start_time = time.time()
                exps, logs1 = algo.collect_experiences()
                logs2 = algo.update_parameters(exps)
                logs = {**logs1, **logs2}
                update_end_time = time.time()
                num_frames += logs["num_frames"]
                student_return_avg.append(self.synthesize(logs["reshaped_return_per_episode"])["mean"])
                update += 1
                print(update, end=",")
                
                if self.args.historical_averaging:
                    teacher_hist_models.append(md)
                    md_index = np.random.choice(range(len(self.student_hist_models)),1)[0]
                    md = copy.deepcopy(self.student_hist_models[md_index])

                if update % self.args.log_interval == 0 and False:
                    fps = logs["num_frames"]/(update_end_time - update_start_time)
                    duration = int(time.time() - start_time)
                    return_per_episode = utils.synthesize(logs["return_per_episode"])
                    rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
                    header = ["update", "frames", "FPS", "duration"]
                    data = [update, num_frames, fps, duration]
                    header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                    data += rreturn_per_episode.values()
                    header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                    data += num_frames_per_episode.values()
                    header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                    data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

                    txt_logger.info(
                        "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                        .format(*data))

                    header += ["return_" + key for key in return_per_episode.keys()]
                    data += return_per_episode.values()

                    if status["num_frames"] == 0:
                        csv_logger.writerow(header)
                    csv_logger.writerow(data)
                    csv_file.flush()

                    for field, value in zip(header, data):
                        tb_writer.add_scalar(field, value, num_frames)
            reward = max([0,self._reward()-numpy.average(student_return_avg)])
            #logs = {**logs1, **logs2}
            #rreturn_per_episode = self.synthesize(logs["reshaped_return_per_episode"])
            #print(rreturn_per_episode)
        return obs, reward, done, {"agent_pos": self.agent_pos}

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        if self.end_pos is not None:
            self.put_obj(Goal(), self.end_pos[0], self.end_pos[1])

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

    def synthesize(self, array):
        d = collections.OrderedDict()
        d["mean"] = numpy.mean(array)
        d["std"] = numpy.std(array)
        d["min"] = numpy.amin(array)
        d["max"] = numpy.amax(array)
        return d

class DoorKeyEnv5x5(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)

class TeacherDoorKeyEnv5x5(TeacherDoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)
register(
    id='MiniGrid-TeacherDoorKey-5x5-v0',
    entry_point='gym_minigrid.envs:TeacherDoorKeyEnv5x5'
)

class TeacherDoorKeyEnv10x10(TeacherDoorKeyEnv):
    def __init__(self):
        super().__init__(size=10)
register(
    id='MiniGrid-TeacherDoorKey-10x10-v0',
    entry_point='gym_minigrid.envs:TeacherDoorKeyEnv10x10'
)

class DoorKeyEnv6x6(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=6)

class DoorKeyEnv16x16(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-DoorKey-5x5-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv5x5'
)

register(
    id='MiniGrid-DoorKey-6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv6x6'
)

register(
    id='MiniGrid-DoorKey-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv'
)

register(
    id='MiniGrid-DoorKey-16x16-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv16x16'
)
