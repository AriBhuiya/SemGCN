#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --mem=34G
#SBATCH --gpus-per-node=1
#SBATCH --output=output.%j.test.out
#SBATCH --mail-user=abhuiya1@sheffieldac.uk
#SBATCH --time=10:00:00

module load Anaconda3/5.3.0

module load cuDNN/7.6.4.38-gcccuda-2019b
source activate pytorch

python main_gcn.py --non_local --epochs 30
