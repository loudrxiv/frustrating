#!/usr/bin/env bash

#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="[M46|M40]"
#SBATCH --job-name=training_baseline
#SBATCH --output=train/%j.out
#SBATCH --error=train/%j.err
#SBATCH --exclude=g023
#SBATCH --mem=220G
date;hostname;id
printf "========================\n"

# Adjust to your desired line number
desired_line=$SLURM_ARRAY_TASK_ID

# Use sed to extract the specific line
line=$(sed -n "${desired_line}p" train_array.txt)

# Split the line into an array using ',' as a delimiter
IFS=',' read -r -a hyperparameters <<< "$line"

# Print the results
tf=${hyperparameters[0]}
target=${hyperparameters[1]}

echo "TF: $tf"
echo "Target: $target"
printf "\n"

# Get ready to saved the log
ROOT=$1
LOGS_TRAIN="${ROOT}/logs/train"
mkdir -p $LOGS_TRAIN

# (1) Activate conda
echo 'activating conda environment'
source activate genomic_tools
which python

# (2) Point to the training file we need
train_file=${ROOT}/3_train_and_test_models/Baseline/train.py
echo 'train file: ' ${train_file}

# (3) Now run a single training job 
echo 'running script'
python train.py "$tf" "$target" > "$LOGS_TRAIN/Baseline_${tf}_${target}.log"