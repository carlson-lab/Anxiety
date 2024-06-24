#!/bin/bash
#SBATCH -p scavenger-gpu
#SBATCH --account=carlsonlab
#SBATCH --job-name=unbal_321_net_anx_kfold_%A_%a
#SBATCH -o /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/batch_job_files/unbal_321_net_kfold_%a.out
#SBATCH -e /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/batch_job_files/unbal_321_net_kfold_%a.err
#SBATCH --array 2
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

srun singularity exec -B /work/mk423/ --nv /work/mk423/cpne_anxiety.simg python /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/unbal_3_to_1_net.py

