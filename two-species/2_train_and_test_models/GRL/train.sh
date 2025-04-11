#!/usr/bin/env bash

#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="[M46|M40|M24|M20|M16]"
#SBATCH --job-name=training_grl
#SBATCH --output=train/%j.out
#SBATCH --error=train/%j.err
#SBATCH --exclude=g023
#SBATCH --mem=180G
date;hostname;id
printf "========================\n"

export TF_CPP_MIN_LOG_LEVEL=1

# Adjust to your desired line number
desired_line=$SLURM_ARRAY_TASK_ID

# Use sed to extract the specific line
line=$(sed -n "${desired_line}p" train_array.txt)

# Split the line into an array using ',' as a delimiter
IFS=',' read -r -a hyperparameters <<< "$line"

# Print the results
tf=${hyperparameters[0]}
source=${hyperparameters[1]}
run=${hyperparameters[2]}
lambda_=${hyperparameters[3]}

echo "TF: $tf"
echo "Source: $source"
echo "Run: $run"
echo "Lambda: $lambda_"
printf "\n"

# Get ready to save the log
ROOT=$1
LOG_ROOT=$2

TRAIN_LOGS="${LOG_ROOT}/train"
mkdir -p "$TRAIN_LOGS"

# (1) Activate conda
echo 'activating conda environment'
source activate tensorflow_A2
which python

# (2) Point to the training file we need
train_file=${ROOT}/2_train_and_test_models/GRL/train.py
echo 'train file: ' ${train_file}

# (3) Now run a single training job 
echo 'running script'
python train.py "$tf" "$source" "$run" "$lambda_" > "$TRAIN_LOGS/GRL_tf-${tf}_source-${source}_run-${run}.log"