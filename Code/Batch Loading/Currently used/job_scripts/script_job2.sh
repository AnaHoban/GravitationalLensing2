#!/bin/sh 
#SBATCH --job-name=job6
#SBATCH --account=rrg-kyi
#SBATCH --time=25:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8000M
#SBATCH --output=%x-%j.out
python job2.py
