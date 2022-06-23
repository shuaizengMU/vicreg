#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
#SBATCH -A xulab-gpu
#SBATCH -p gpu4
#SBATCH -t 2-00:00
#SBATCH -N 1  # number of nodes
#SBATCH -n 20  # number of cores (AKA tasks)
#SBATCH --mem=64G
#SBATCH --gres gpu:0
#
## labels and outputs
#SBATCH -J IMG  # give the job a custom name
#SBATCH -o ./stdout/imagenet_mini_generator.out  # give the job output a custom name
#
#-------------------------------------------------------------------------------

nvidia-smi

# Activate venv
. ~/zengs/venv_hub/venv_torch_38/bin/activate

python dataset_imagenet_mini_generator.py