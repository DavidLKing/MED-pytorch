#!/usr/bin/env bash

#SBATCH --time=20:00:00
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16gb
#SBATCH --job-name=anderson_exp_string
#SBATCH --mail-type=ALL
#SBATCH --mail-user=osu8940
#SBATCH --output=exp_string.%j
#SBATCH --account=PZS1006

module load python/3.6-conda5.2
module load cuda

cd /home/david/bin/git/MED-pytorch

source activate med

python main.py --config configs/artif-final-0.50-50/config.anderson_aab.yml
