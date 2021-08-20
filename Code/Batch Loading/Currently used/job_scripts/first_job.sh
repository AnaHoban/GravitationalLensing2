#!/bin/sh 
#SBATCH --job-name=job4
#SBATCH --account=rrg-kyi
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8000M
#SBATCH --output=%x-%j.out
python job1.py
