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
#SBATCH --gres gpu:2
#
## labels and outputs
#SBATCH -J IMG100  # give the job a custom name
#SBATCH -o ./stdout/imagenet-samemini-100.out  # give the job output a custom name
#
#-------------------------------------------------------------------------------

# Activate venv
. ~/zengs/venv_hub/venv_torch_38/bin/activate

# check gpu
nvidia-smi

# Run code
echo "run ssl imagenet-samemini-100"
python -m torch.distributed.launch --nproc_per_node=2 main_vicreg_original.py  --data-dir  ./data/ILSVRC2012/imagenet-samemini-100 --exp-dir ./exp_hub/imagenet-samemini-100 --epochs 50 --batch-size 128 --base-lr 0.8

# evaluation
echo "run evaluation imagenet-samemini-100"
python evaluate_original.py --data-dir ./data/ILSVRC2012/imagenet-samemini-100 --pretrained ./exp_hub/imagenet-samemini-100/resnet50.pth --exp-dir ./exp_hub/imagenet-samemini-100 --lr-head 0.02 --epochs 100