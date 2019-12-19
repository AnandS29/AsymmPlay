Setup:
Install everything locally.

Test:
python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --model nt2 --teacher_algo a2c --student_algo ppo -t 1 3 -e 3 1

python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model t2 --teacher_algo a2c --student_algo ppo -t 1 3 -e 3 1

python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model t_ha2 --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 -t 1 3 -e 3 1

python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model t_ha_ex_1 --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 -t 3 1 -e 1 3 --sampling_strategy exponential
