#!/bin/sh 
#SBATCH --job-name=create_dataset
#SBATCH --account=rrg-kyi
#SBATCH --time=30:00:00
#SBATCH --mem=8000M
#SBATCH --output=%x-%j.out
python download_tiles.py
