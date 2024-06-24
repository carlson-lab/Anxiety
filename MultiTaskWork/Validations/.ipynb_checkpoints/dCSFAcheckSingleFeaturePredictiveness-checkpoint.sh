#!/bin/bash
#SBATCH -p carlsonlab-gpu
#SBATCH --account=carlsonlab
#SBATCH --job-name=anx_single_feature
#SBATCH -o /hpc/home/mk423/AnxSingleFeatureTest.out
#SBATCH -e /hpc/home/mk423/AnxSingleFeatureTest.err
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

srun singularity exec -B /work/mk423/ --nv /work/mk423/cpne_anxiety.simg python /hpc/home/mk423/Anxiety/MultiTaskWork/Validations/dCSFAcheckSingleFeaturePredictiveness.py

