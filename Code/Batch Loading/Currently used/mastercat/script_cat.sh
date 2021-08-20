#!/bin/bash 
#SBATCH --job-name=create_noisy_cutouts
#SBATCH --account=def-sfabbro
#SBATCH --time=25:00:00
#SBATCH --mem=8000M
source $HOME/lensing/bin/activate
python ../cutouts/creating_candidates_cutouts.py
