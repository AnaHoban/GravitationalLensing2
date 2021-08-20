#!/bin/sh
#SBATCH --job-name=12epochs-12tiles
#SBATCH --account=rrg-kyi
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8000M
#SBATCH --output=%x-%j.out
python job2_128.py
