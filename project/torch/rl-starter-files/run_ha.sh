for i in {1..10}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model 5x5_0.05_exp_inter_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.05 -t 1 3 --sampling_strategy exponential -e 3 1 --frames_per_proc 10 --seed $i; done

for i in {1..10}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model 5x5_0.05_exp_inter_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.05 -t 1 3 --sampling_strategy exponential --rand_goal --seed $i; done

for i in {1..10}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model 5x5_0.05_exp_inter_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.05 -t 1 3 --sampling_strategy exponential --rand_goal --seed $i; done

python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model 5x5_0.05_exp_inter_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 -t 1 3 --sampling_strategy exponential --rand_goal --frames_teacher 20

python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model t2 --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 -t 5 1 --sampling_strategy exponential --rand_goal --frames_teacher 20
