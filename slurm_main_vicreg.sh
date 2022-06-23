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
#SBATCH -J main_vicreg_imagenet  # give the job a custom name
#SBATCH -o ./stdout/results-%j.out  # give the job output a custom name
#
#-------------------------------------------------------------------------------

# Activate venv
. ~/zengs/venv_hub/venv_torch_38/bin/activate

# check gpu
nvidia-smi

# Run code
echo "run ssl"
# imagenet-100-samemini
# python -m torch.distributed.launch --nproc_per_node=2 main_vicreg_original.py  --data_dir  ./data/ILSVRC2012/imagenet-100-samemini --exp_dir ./exp_hub/imagenet-100-samemini --epochs 50 --batch_size 128

# imagenet
python -m torch.distributed.launch --nproc_per_node=2 main_vicreg_original.py  --data-dir  ./data/ILSVRC2012/imagenet --exp-dir ./exp_hub/imagenet  --arch resnet50 --epochs 100 --batch-size 128 --base-lr 0.8