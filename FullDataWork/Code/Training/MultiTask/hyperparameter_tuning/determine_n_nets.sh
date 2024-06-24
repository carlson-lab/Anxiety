#!/bin/bash
#SBATCH -p scavenger-gpu
#SBATCH --account=carlsonlab
#SBATCH --job-name=anx_n_net_%A_%a
#SBATCH -o /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/batch_job_files/anx_n_net_%a.out
#SBATCH -e /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/batch_job_files/anx_n_net_%a.err
#SBATCH --array 0-28
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

srun singularity exec -B /work/mk423/ --nv /work/mk423/cpne_anxiety.simg python /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/determine_n_nets.py

