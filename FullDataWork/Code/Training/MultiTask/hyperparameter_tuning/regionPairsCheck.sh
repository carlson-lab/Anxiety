#!/bin/bash
#SBATCH -p scavenger-gpu
#SBATCH --account=carlsonlab
#SBATCH --job-name=brPair_check_%A_%a_LR
#SBATCH -o /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/MultiTask/regionPair_batch_files/brPair_check_%a_LR.out
#SBATCH -e /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/MultiTask/regionPair_batch_files/brPair_check_%a_LR.err
#SBATCH --array 0-27
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

srun singularity exec -B /work/mk423/ --nv /work/mk423/cpne_anxiety.simg python /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/MultiTask/hyperparameter_tuning/regionPairsCheck.py

