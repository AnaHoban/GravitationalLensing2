#!/bin/sh 
#SBATCH --job-name=candidate_training
#SBATCH --account=rrg-kyi
#SBATCH --time=3-00:00
#SBATCH --mem=1000G
source $HOME/umap/bin/activate
python create_tf_dataset.py
