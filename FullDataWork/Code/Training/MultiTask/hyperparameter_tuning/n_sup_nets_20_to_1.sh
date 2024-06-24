#!/bin/bash
#SBATCH -p carlsonlab-gpu
#SBATCH --account=carlsonlab
#SBATCH --job-name=20_sup_net_%A_%a
#SBATCH -o /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/MultiTask/batch_job_files/20_sup_net_%a.out
#SBATCH -e /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/MultiTask/batch_job_files/20_sup_net_%a.err
#SBATCH --array 1-4
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

srun singularity exec -B /work/mk423/ --nv /work/mk423/cpne_anxiety.simg python /hpc/home/mk423/Anxiety/FullDataWork/Code/Training/MultiTask/n_sup_nets_20_to_1.py

