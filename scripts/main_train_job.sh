#!/bin/bash
#SBATCH --job-name=model_tr_classification   # Job name
#SBATCH --output=classification_train.out       # Standard output
#SBATCH --error=classification_train.err        # Standard error
#SBATCH -c 8                              # Request 4 CPU cores
#SBATCH --mem=32G                         # Request 32 GB of memory
#SBATCH --time=48:00:00                   # Set a runtime limit of 24 hours

# Activate your Conda environment
source /data/.../miniforge3/etc/profile.d/conda.sh 
conda activate yourenv

# Navigate to the script's directory 
cd ../scripts/

# Run the Python script
python main_train.py

