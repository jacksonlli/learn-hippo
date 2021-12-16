#!/bin/bash
#SBATCH --account=rrg-bashivan_cpu
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=10
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

pip install --no-index torch==1.6.0
pip install --no-index matplotlib
pip install --no-index seaborn
pip install --no-index scikit-learn
pip install --no-index /home/daiglema/scratch/rl_poly/learn-hippo/packages/bidict-0.18.2-py2.py3-none-any.whl


exp_name=vary-training-penalty

n_def_tps=0
similarity_max=.9
similarity_min=0
sup_epoch=600
n_epoch=1000
penalty_random=0
attach_cond=0
cmpt=.8
dict_len=2
noRL=0
enc_size=8
n_param=8
n_branch=4

for subj_id in {0..15}
do
   for penalty in 0 4
   do
       for def_prob in .25
       do
           sbatch train-model.sh ${exp_name} \
               ${subj_id} ${penalty} ${n_epoch} ${sup_epoch} \
               ${similarity_max} ${similarity_min} \
               ${penalty_random} ${def_prob} ${n_def_tps} ${cmpt} ${attach_cond} \
               ${enc_size} ${dict_len} ${noRL} ${n_param} ${n_branch}
       done
   done
done
