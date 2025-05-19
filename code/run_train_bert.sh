#!/bin/bash
#SBATCH -p GPUExtended
#SBATCH -N 1                 # Use 1 node
#SBATCH --gres=gpu:1         # Request 1 GPU
#SBATCH -o slurm.%N.%j.out   # Standard output log
#SBATCH -e slurm.%N.%j.err   # Error log

# Load Conda
source ~/.bashrc
conda activate biasintransformers  # Ensure environment is activated

# Run the training script
python train_bert_sentence.py
