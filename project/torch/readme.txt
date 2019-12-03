Setup:
Install everything locally.


Run the following (you may need to delete files in storage folder):
python3 -m scripts.train --algo ppo --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 16 --model test_no_teach
python3 -m scripts.train --algo ppo --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 16 --model test_teach --teach

Notice that the avg reward for evaluation is higher for the teaching than with the not teaching. The different is that teaching has the following steps:
  1. Self play iterations
  2. Training on environment where [3,3] is goal
  3. Evaluate on environment where [3,1] is goal

When not using teaching, we have just steps 2, 3.

This could show that warm starting with self-play makes policies more generalizable.

Notes:
  Teacher uses a2c, student uses PPO

Possible next steps:
  Harder Environments
  Visualization of movement in evaluation step
  Historical policy averaging
  Compare selfplay with just random walks of a teacher - it performs better from my tests (though need to formalize a bit more)

#0.71,0.94
Old test (may or may not work):
python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --model nt_11_21_3_1 --teacher_algo a2c --student_algo ppo

python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model t_11_21_3_1 --teacher_algo a2c --student_algo ppo

python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model t_ha_11_21_3_1 --teacher_algo a2c --student_algo ppo --historical_averaging 0.2
Newer tests:

python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --model nt_11_27_1_3_o_2 --teacher_algo a2c --student_algo ppo --seed 2

python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model t_11_27_1_3_o_2 --teacher_algo a2c --student_algo ppo --seed 2

python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model t_ha_11_27_1_3_o_2 --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --seed 2

python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 10 --t_iters 0 --s_iters_per_teaching 5 --model eval_test_1 --teacher_algo a2c --student_algo ppo
