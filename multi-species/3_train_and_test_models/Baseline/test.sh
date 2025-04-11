#!/usr/bin/env bash

#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="[M46|M40]"
#SBATCH --job-name=testing_baseline
#SBATCH --output=test/%j.out
#SBATCH --error=test/%j.err
#SBATCH --exclude=g023
#SBATCH --mem=220G
date;hostname;id
printf "========================\n"

# Adjust to your desired line number
desired_line=$SLURM_ARRAY_TASK_ID

# Use sed to extract the specific line
line=$(sed -n "${desired_line}p" test_array.txt)

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
LOG_TESTS="${ROOT}/logs/test"
mkdir -p $LOG_TESTS

# (1) Activate conda
echo 'activating conda environment'
source activate genomic_tools
which python

# (2) Point to the training file we need
test_file=${ROOT}/3_train_and_test_models/Baseline/test.py
echo 'test file: ' ${test_file}

# (3) Now run a single training job 
echo 'running script'
python test.py "$tf" "$target" > "$LOG_TESTS/Baseline_${tf}_${target}.log"