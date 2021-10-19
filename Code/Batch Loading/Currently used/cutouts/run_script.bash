#!/bin/sh 
#SBATCH --job-name=download_tiles
#SBATCH --account=rrg-kyi
#SBATCH --time=48:00:00
#SBATCH --mem=8000M
#SBATCH --output=%x-%j.out
python download_tiles.py
