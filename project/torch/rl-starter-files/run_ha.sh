for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model teaching_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0 --sampling_strategy exponential -t 5 1 --rand_goal --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model teaching_ha_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 5 1 --rand_goal --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 0 --s_iters_per_teaching 10 --model no_teaching_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0 --sampling_strategy exponential -t 5 1 --rand_goal --seed $i; done
