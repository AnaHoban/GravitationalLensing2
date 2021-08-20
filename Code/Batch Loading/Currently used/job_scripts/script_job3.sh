#!/bin/sh 
#SBATCH --job-name=job8-unbalanced_tiles
#SBATCH --account=def-sfabbro
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8000M
source $HOME/lensing/bin/activate
python job3.py