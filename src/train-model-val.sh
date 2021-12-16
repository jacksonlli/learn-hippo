#!/bin/bash
#SBATCH -t 19:55:00
#SBATCH -c 1
#SBATCH --mem-per-cpu 4G

#SBATCH --job-name=lcarnn
#SBATCH --output slurm_log/lcarnn-%j.log

#module load anaconda

DATADIR=/home/daiglema/scratch/rl_poly/learn-hippo/log

echo $(date)

srun python -u train-sl.py --exp_name ${1} --subj_id ${2} --penalty ${3}  \
    --n_epoch ${4} --sup_epoch ${5} --similarity_max ${6} --similarity_min ${7} \
    --penalty_random ${8} --cmpt ${9} --attach_cond ${10} \
    --enc_size ${11} --dict_len ${12} --noRL ${13} --n_param ${14} --n_branch ${15}\
    --log_root $DATADIR

echo $(date)
