from gym_minigrid.minigrid import *
from gym_minigrid.register import register

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
        #print("Stuff")
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.grid.set(splitIdx, doorIdx, Door('yellow', is_locked=True))

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

class DoorKeyEnv5x5(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)

class DoorKeyEnv6x6(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=6)

class DoorKeyEnv16x16(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=16)

class StudentEnv(DoorKeyEnv):
    def __init__(self, size=5):
        self.goal_pos = None
        self.teacher_step_count = None
        self.size = size

    def setup(self):
        super().__init__(size=self.size)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.grid.set(self.goal_pos[0], self.goal_pos[1], Goal())

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.grid.set(splitIdx, doorIdx, Door('yellow', is_locked=True))

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

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
            if fwd_cell != None and fwd_cell.type == 'goal':
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
            pass

        else:
            assert False, "unknown action"

        obs = self.gen_obs()

        if self.step_count >= self.max_steps:
            done = True
            return obs, self.max_steps - self.teacher_step_count, done, {}


        return obs, (0 if not done else self.step_count), done, {}

class TTestEnv(DoorKeyEnv):
    def __init__(self, size=5):
        super().__init__(size=size)
    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

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
            if fwd_cell != None and fwd_cell.type == 'goal':
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
            done = True

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

class TeacherEnv(DoorKeyEnv):
    def __init__(self, size=5):
        super().__init__(size=size)
        self.action_space = spaces.Discrete(len(self.actions))
        self.gamma = 1
        self.observation_space = spaces.Box(low=0, high=100, shape=(7*7*3,))
        self.initial_pos = self.agent_pos[:]

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

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
            if False and fwd_cell != None and fwd_cell.type == 'goal':
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
            done = True
            #print("Done")
        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, (0 if done else self.step_count), done, {"start": self.initial_pos, "end": self.agent_pos[:], "teacher_step_count": self.step_count}

class TTestEnv16x16(TTestEnv):
    def __init__(self):
        super().__init__(size=5)

class StudentEnv5x5(StudentEnv):
    def __init__(self):
        super().__init__(size=16)

class TeacherEnv5x5(TeacherEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-TTestEnv-16x16-v0',
    entry_point='gym_minigrid.envs:TTestEnv16x16'
)

register(
    id='MiniGrid-StudentEnv-5x5-v0',
    entry_point='gym_minigrid.envs:StudentEnv5x5'
)

register(
    id='MiniGrid-TeacherEnv-5x5-v0',
    entry_point='gym_minigrid.envs:TeacherEnv5x5'
)

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
