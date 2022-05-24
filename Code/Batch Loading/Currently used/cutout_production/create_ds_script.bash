#!/bin/sh 
#SBATCH --job-name=create_dataset
#SBATCH --account=def-sfabbro
#SBATCH --time=4:00:00
#SBATCH --mem=8000M
#SBATCH --output=%x-%j.out
source $HOME/umap/bin/activate
python create_trainingdataset.py
