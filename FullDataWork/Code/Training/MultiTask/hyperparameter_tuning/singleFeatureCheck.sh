#!/bin/bash
#SBATCH -p scavenger-gpu #carlsonlab-gpu or scavenger-gpu or common, or scavenger
#SBATCH --account=carlsonlab
#SBATCH --job-name=sf_check_%A_%a_LR
#SBATCH -o /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/MultiTask/singleFeature_batch_files/sf_check_%a_cCSFA.out
#SBATCH -e /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/MultiTask/singleFeature_batch_files/sf_check_%a_cCSFA.err
#SBATCH --array 2,3 #0-91
#SBATCH -c 12 #number of cores - go fast
#SBATCH --mem=80G #amount of memory - lots of data
#SBATCH --gres=gpu:1 #number of gpus (always just 1 or 0) - fast for neural networks

srun singularity exec -B /work/mk423/ --nv /work/mk423/cpne_anxiety.simg python /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/MultiTask/hyperparameter_tuning/singleFeatureCheck.py
#srun singularity exec -B {path to your work directory} --nv {path to your singularity container} {path to python file}

# module load pythonXXX
# bash command to set python path to anaconda path
# run file

