!/bin/sh 
#SBATCH --job-name=candidate_training
#SBATCH --account=rrg-kyi
#SBATCH --time=55:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8000M
source $HOME/lensing/bin/activate
python test_job.py
