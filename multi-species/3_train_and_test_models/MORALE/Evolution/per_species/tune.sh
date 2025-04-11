#!/usr/bin/env bash

#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="[M46|M40]"
#SBATCH --job-name=tune_evops
#SBATCH --output=tune/%j.out
#SBATCH --error=tune/%j.err
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
_lambda=${hyperparameters[2]}
match_mean=${hyperparameters[3]}
holdout=${hyperparameters[4]}

echo "TF: $tf"
echo "Target: $target"
echo "Lambda: $_lambda"
echo "Match mean: $match_mean"
echo "Holdout index: $holdout"
printf "\n"

# Get ready to saved the log
ROOT=$1
TUNE_LOGS="${ROOT}/logs/tune"
mkdir -p $TUNE_LOGS

# (1) Activate conda
echo 'activating conda environment'
source activate pytorch
which python

# (2) Point to the tuning file we need
echo 'running script'
python tune.py "$tf" "$target" "$_lambda" "$match_mean" "$holdout" > "$TUNE_LOGS/EvoPS_tf-${tf}_target-${target}_lamb-${_lambda}_mean-${match_mean}_holdout-${holdout}.log"