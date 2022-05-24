#!/bin/sh 
#SBATCH --job-name=training_test
#SBATCH --account=rrg-kyi
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00
#SBATCH --mem=50G
#SBATCH --task-per-node=1
source $HOME/umap/bin/activate
python autoencoder_training.py