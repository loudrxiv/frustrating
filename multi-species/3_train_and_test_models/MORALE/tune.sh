#!/usr/bin/env bash

#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="[M46|M40]"
#SBATCH --job-name=tuning_morale
#SBATCH --output=tune/%j.out
#SBATCH --error=tune/%j.err
#SBATCH --exclude=g023
#SBATCH --mem=220G
date;hostname;id
printf "========================\n"

# Adjust to your desired line number
desired_line=$SLURM_ARRAY_TASK_ID

# Use sed to extract the specific line
line=$(sed -n "${desired_line}p" tune_array.txt)

# Split the line into an array using ',' as a delimiter
IFS=',' read -r -a hyperparameters <<< "$line"

# Print the results
tf=${hyperparameters[0]}
target=${hyperparameters[1]}
_match_mean=${hyperparameters[2]}
_lambda=${hyperparameters[3]}

echo "TF: $tf"
echo "Target: $target"
echo "Match Mean: $_match_mean"
echo "Lambda: $_lambda"
printf "\n"

# Get ready to saved the log
ROOT=$1
TUNE_LOGS="${ROOT}/logs/tune"
mkdir -p $TUNE_LOGS

# (1) Activate conda
echo 'activating conda environment'
source activate genomic_tools
which python

# (2) Point to the tuning file we need
tune_file=${ROOT}/2_train_and_test_models/MORALE/tune.py
echo 'tune file: ' ${tune_file}

# (3) Now run a single tuning job 
echo 'running script'
python tune.py "$tf" "$target" "$_match_mean" "$_lambda" > "$TUNE_LOGS/MORALE_tf-${tf}_target-${target}_mean-${_match_mean}_lambda-${_lambda}.log"