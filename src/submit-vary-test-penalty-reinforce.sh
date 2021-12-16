#!/bin/bash
#SBATCH --account=rrg-bashivan
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M
#SBATCH --output=slurm/slurm-%j.out

# ##S#B#A#TC#H -#-g#res=#gpu:v100:#1

echo Running on $HOSTNAME
nvidia-smi
date

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
echo $SLURM_TMPDIR
source $SLURM_TMPDIR/env/bin/activate

#virtualenv --no-download /home/daiglema/scratch/rl_poly/env
#echo /home/daiglema/scratch/rl_poly/env/bin/activate
#source /home/daiglema/scratch/rl_poly/env/bin/activate

pip install --no-index torch==1.6.0
pip install --no-index matplotlib
pip install --no-index seaborn
pip install --no-index scikit-learn
pip install --no-index /home/daiglema/scratch/rl_poly/learn-hippo/packages/bidict-0.18.2-py2.py3-none-any.whl

DATADIR=/home/daiglema/scratch/rl_poly/learn-hippo/log
similarity_max=.9
similarity_min=0
sup_epoch=600
n_epoch=1000
penalty_random=1
attach_cond=0
cmpt=.8
dict_len=2
noRL=0
enc_size=16
n_param=16
n_branch=4

exp_name=vary-test-penalty-loss-reinforce
gamma=0.99
is_a2c=0



for subj_id in {0..2}
do
   for penalty in 4
   do
       echo $(date)
       srun python -u train-sl.py --exp_name ${exp_name} --subj_id ${subj_id} --penalty ${penalty}  \
           --n_epoch ${n_epoch} --sup_epoch ${sup_epoch} --similarity_max ${similarity_max} --similarity_min ${similarity_min} \
           --penalty_random ${penalty_random} --cmpt ${cmpt} --attach_cond ${attach_cond} \
           --enc_size ${enc_size} --dict_len ${dict_len} --noRL ${noRL} --n_param ${n_param} --n_branch ${n_branch}\
           --log_root $DATADIR --gamma ${gamma} --is_a2c ${is_a2c}
       echo $(date)
   done
done













