#!/bin/bash
#SBATCH -p carlsonlab-gpu
#SBATCH --account=mk423
#SBATCH --job-name=singleFeatureDCSFA
#SBATCH -o /hpc/home/mk423/Anxiety/MultiTaskWork/single-feature-dcsfa.out
#SBATCH -e /hpc/home/mk423/Anxiety/MultiTaskWork/single-feature-dcsfa.err
#SBATCH --exclude=dcc-mastatlab-gpu-01,dcc-tdunn-gpu-[01-02],dcc-dsplus-gpu-[01-04],dcc-pbenfeylab-gpu-02,dcc-youlab-gpu-[01-57],dcc-gehmlab-gpu-[01-04],dcc-rekerlab-gpu-01,dcc-carlsonlab-gpu-[01-03],dcc-dhvi-gpu-[01-04],dcc-biostat-gpu-[01-02],dcc-viplab-gpu-01
#SBATCH -c 2
#SBATCH --mem=80G
#SBATCH --gres=gpu:1

srun singularity exec -B /work/mk423/ --nv /work/mk423/cpne_anxiety.simg python /hpc/home/mk423/Anxiety/MultiTaskWork/Code/dCSFAcheckSingleFeaturesPrediction.py

