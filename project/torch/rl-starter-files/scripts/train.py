import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX
import sys
from torch_ac.utils import ParallelEnv
import utils
from model import ACModel
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import copy

# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--teacher_algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--student_algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
# Sandy: kinda implemented not sure if right; probably wrong

parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--teacher_model", default=None,
                    help="name of the teacher model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--eval-interval", type=int, default=1,
                    help="number of updates between two evals (default: 1)")
parser.add_argument("--save-interval", type=int, default=1,
                    help="number of updates between two saves (default: 1, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--t_iters", type=int, default=0,
                    help="teaching iterations (default: 5)")
parser.add_argument("--s_iters_per_teaching", type=int, default=5,
                    help="student iterations per teaching iteration (default: 5)")
parser.add_argument("--nt_iters", type=int, default=0,
                    help="non-teaching iterations (default: 5)")

parser.add_argument("--rand_goal", action="store_true", default=False,
                    help="use random goals for evaluation")
# Sandy: not yet implemented and need to be implemented

parser.add_argument('-e','--eval_goal', nargs='+', type=int, default=[3,1], help='evaluation goal', required=False)
parser.add_argument('-t','--train_goal', nargs='+', type=int, default=[1,3], help='training goal', required=True)

parser.add_argument("--historical_averaging", type=float, default=0,
                    help="probability for historical averaging (default: 0)")
parser.add_argument("--sampling_strategy", required=False, default="uniform",
                    help="sampling strategy for historical averaging (default: uniform)")
parser.add_argument("--intra", action="store_true", default=False,
                    help="use intra historical averaging")

## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--frames_teacher", type=int, default=10,
                    help="number of frames for teacher before update (default: 10)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

# Arguments for eval
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--episodes", type=int, default=100, help="number of episodes of evaluation (default: 100)")

args = parser.parse_args()

print("Memory ",args.recurrence)
args.mem = args.recurrence > 1

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
#default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"
default_model_name = "{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#txt_logger.info(f"Device: {device}\n")
txt_logger.info("Device: {device}\n")

# Load environments
envs = []
s = 10
for i in range(args.procs):
    env = utils.make_env(args.env, args.seed)
    env.is_teaching = False
    env.end_pos = args.train_goal
    envs.append(env)
txt_logger.info("Environments loaded\n")

teacher_env = utils.make_env(args.env, args.seed)
teacher_env.is_teaching = True

# Load training status

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# Load model

acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
# historical_models = [acmodel]
if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])
acmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

if "optimizer_state" in status and False:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

student_hist_models = [acmodel]
teacher_env.student_hist_models = student_hist_models
# teacher_env.algo = teacher_algo # Sandy added
teacher_env.args = args
teacher_env.model_dir = model_dir
teacher_env.preprocess_obss = preprocess_obss

def run_eval():
    envs = []
    for i in range(8):
        # env = utils.make_env(args.env, args.seed + 10000 * i)
        env = utils.make_env(args.env, args.seed)
        env.is_teaching = False
        if args.rand_goal:
            pos = env._rand_pos(0, env.width, 0, env.height)
            while env.grid.get(*pos) is None:
                pos = env._rand_pos(0, env.width, 0, env.height)
            eval = list(pos)
        else:
            eval = args.eval_goal
        env.end_pos = eval
        envs.append(env)
    env = ParallelEnv(envs)

    # Load agent

    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir, device, args.argmax, args.procs)

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)
    positions = []
    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, dones, infos = env.step(actions)
        positions.extend([info["agent_pos"] for info in infos])
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("Eval: F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))
    return return_per_episode

# Sampling distribution for historical historical averaging
def sampling_dist(n,strategy=args.sampling_strategy):
    if strategy == "uniform":
        return np.ones(n)/n
    elif strategy == "exponential":
        prob = np.array([1.2 ** i for i in range(n)], dtype=np.float)
        prob = prob/np.sum(prob)
        return prob
# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()
# python3 -m scripts.train --algo ppo --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000
j = 0
if args.t_iters > 0:
    teach_acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    teach_acmodel.to(device)
    teacher_hist_models = [teach_acmodel]

    print("Starting to teach")
    # if np.random.random() < args.historical_averaging and not args.intra:
    #     md = copy.deepcopy(teach_acmodel)
    # else:
    #     md = teach_acmodel
    md = teach_acmodel

    while j < args.t_iters:
        # Add options for teacher algo
        if args.teacher_algo == "a2c":
            algo_teacher = torch_ac.A2CAlgo([teacher_env], md, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
        elif args.teacher_algo == "ppo":
            algo_teacher = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
        else:
            raise ValueError("Incorrect algorithm name: {}".format(args.algo))
        # END Add options for teacher algo

        algo_teacher.args = args

        if args.intra:
            algo_teacher.historical_models = teacher_hist_models

        exps, logs1 = algo_teacher.collect_experiences()
        logs2 = algo_teacher.update_parameters(exps)
        j += 1

        #run_eval()

        if not args.intra:
            teacher_hist_models.append(copy.deepcopy(md))
        if np.random.random() < args.historical_averaging and not args.intra:
            md_index = np.random.choice(range(len(teacher_hist_models)),1,p=sampling_dist(len(teacher_hist_models)))[0]
            md = copy.deepcopy(teacher_hist_models[md_index])

        print("Finished teaching iteration ", str(j))

    teacher_env.close()
    print("Done teaching")

acmodel = teacher_env.student_hist_models[-1]

if args.nt_iters > 0:
    update = 0
    envs = []
    for i in range(args.procs):
        env = utils.make_env(args.env, args.seed)
        env.is_teaching = False
        env.end_pos = args.train_goal
        envs.append(env)

    # Add options for student algo
    if args.student_algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, md, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.student_algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, md, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                        args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                        args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))
    # END

    algo.args = args
    #while (num_frames < args.frames) and update < 28:
    while update < args.nt_iters:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()
        num_frames += logs["num_frames"]
        update += 1
        # Print logs


                # Save status

        if (args.save_interval > 0 and update % args.save_interval == 0) or True:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

        if update % args.log_interval == 0:
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

            print("Printing training results ...")
            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            if update % args.eval_interval == 0:
                print("Running eval ...")
                eval_rets = run_eval()

                header += ["eval_return_" + key for key in eval_rets.keys()]
                data += eval_rets.values()

            # header += ["return_" + key for key in eval_rets.keys()]
            # data += eval_rets.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)


# evaluation

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Device: {device}\n")
print("Device: {device}\n")

# Load environments

envs = []
for i in range(1):
    env = utils.make_env(args.env, args.seed + 10000 * i)
    env.is_teaching = False
    env.end_pos = args.eval_goal
    envs.append(env)
env = ParallelEnv(envs)
print("Environments loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir, device, args.argmax, args.procs)
print("Agent loaded\n")

# Initialize logs

logs = {"num_frames_per_episode": [], "return_per_episode": []}

# Run agent

start_time = time.time()

obss = env.reset()

log_done_counter = 0
log_episode_return = torch.zeros(args.procs, device=device)
log_episode_num_frames = torch.zeros(args.procs, device=device)
positions = []
while log_done_counter < args.episodes:
    actions = agent.get_actions(obss)
    obss, rewards, dones, infos = env.step(actions)
    positions.extend([info["agent_pos"] for info in infos])
    agent.analyze_feedbacks(rewards, dones)

    log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
    log_episode_num_frames += torch.ones(args.procs, device=device)

    for i, done in enumerate(dones):
        if done:
            log_done_counter += 1
            logs["return_per_episode"].append(log_episode_return[i].item())
            logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

    mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
    log_episode_return *= mask
    log_episode_num_frames *= mask

end_time = time.time()

# Print logs

num_frames = sum(logs["num_frames_per_episode"])
fps = num_frames/(end_time - start_time)
duration = int(end_time - start_time)
return_per_episode = utils.synthesize(logs["return_per_episode"])
num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
      .format(num_frames, fps, duration,
              *return_per_episode.values(),
              *num_frames_per_episode.values()))

# Print worst episodes

n = args.worst_episodes_to_show
if n > 0:
    print("\n{} worst episodes:".format(n))

    indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    for i in indexes[:n]:
        print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))

# Heatmap
grid = np.zeros((s,s))
positions = [(j,i) for i,j in positions]
counts = Counter(positions)
for i in range(s):
    for j in range(s):
        c = counts[(i,j)]
        grid[i][j] = c/len(positions)
plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.savefig('storage/'+args.model+'/heat_'+str(time.time())+'.png')
print(grid)
