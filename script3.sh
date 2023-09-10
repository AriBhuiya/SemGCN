#!/bin/bash
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --nodes=1
#SBATCH --mem=34G
#SBATCH --gpus-per-node=1
#SBATCH --output=output.%j.test.out
#SBATCH --mail-user=abhuiya1@sheffield.ac.uk
#SBATCH --time=2-10:00:00

module load Anaconda3/5.3.0

module load cuDNN/7.6.4.38-gcccuda-2019b
source activate pytorch

python main_gcn.py --keypoints hr --dropout 0.1
