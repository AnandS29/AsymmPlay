# inter vs intra (same eval and training, diff eval, rand eval)

for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.2_inter_same_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 5 1 -e 5 1 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.2_inter_diff_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 5 1 -e 5 6 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.2_inter_rand_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 5 1 --rand_goal --seed $i; done

for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.2_intra_same_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 5 1 -e 5 1 --seed $i --intra; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.2_intra_diff_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 5 1 -e 5 6 --seed $i --intra; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.2_intra_rand_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 5 1 --rand_goal --seed $i --intra; done

# ha probs uniform (0.05 vs 0.4) (same eval and training, diff eval, rand eval)

for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_uni_0.05_inter_same_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.05 --sampling_strategy uniform -t 5 1 -e 5 1 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_uni_0.05_inter_diff_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.05 --sampling_strategy uniform -t 5 1 -e 5 6 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_uni_0.05_inter_rand_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.05 --sampling_strategy uniform -t 5 1 --rand_goal --seed $i; done

for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_uni_0.4_inter_same_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.4 --sampling_strategy uniform -t 5 1 -e 5 1 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_uni_0.4_inter_diff_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.4 --sampling_strategy uniform -t 5 1 -e 5 6 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_uni_0.4_inter_rand_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.4 --sampling_strategy uniform -t 5 1 --rand_goal --seed $i; done

# ha probs (0.05 vs 0.4) (same eval and training, diff eval, rand eval)

for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.05_inter_same_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.05 --sampling_strategy exponential -t 5 1 -e 5 1 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.05_inter_diff_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.05 --sampling_strategy exponential -t 5 1 -e 5 6 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.05_inter_rand_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.05 --sampling_strategy exponential -t 5 1 --rand_goal --seed $i; done

for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.4_inter_same_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.4 --sampling_strategy exponential -t 5 1 -e 5 1 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.4_inter_diff_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.4 --sampling_strategy exponential -t 5 1 -e 5 6 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.4_inter_rand_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.4 --sampling_strategy exponential -t 5 1 --rand_goal --seed $i; done

#  sampling dist (uni vs exp) (same eval and training, diff eval, rand eval)

for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.2_inter_same_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 5 1 -e 5 1 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.2_inter_diff_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 5 1 -e 5 6 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_exp_0.2_inter_rand_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 5 1 --rand_goal --seed $i; done

for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_uni_0.2_inter_same_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy uniform -t 5 1 -e 5 1 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_uni_0.2_inter_diff_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy uniform -t 5 1 -e 5 6 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-10x10-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 10 --model ha_uni_0.2_inter_rand_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy uniform -t 5 1 --rand_goal --seed $i; done

# inter vs intra for 5x5 (same eval and training, diff eval, rand eval)

for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model easy_ha_exp_0.2_inter_same_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 1 3 -e 1 3 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model easy_ha_exp_0.2_inter_diff_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 1 3 -e 3 1 --seed $i; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model easy_ha_exp_0.2_inter_rand_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 1 3 --rand_goal --seed $i; done

for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model easy_ha_exp_0.2_intra_same_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 1 3 -e 1 3 --seed $i --intra; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model easy_ha_exp_0.2_intra_diff_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 1 3 -e 3 1 --seed $i --intra; done
for i in {1..5}; do python3 -m scripts.train --env MiniGrid-TeacherDoorKey-5x5-v0 --save-interval 10 --frames 80000 --procs 8 --nt_iters 50 --t_iters 10 --s_iters_per_teaching 5 --model easy_ha_exp_0.2_intra_rand_$i --teacher_algo a2c --student_algo ppo --historical_averaging 0.2 --sampling_strategy exponential -t 1 3 --rand_goal --seed $i --intra; done