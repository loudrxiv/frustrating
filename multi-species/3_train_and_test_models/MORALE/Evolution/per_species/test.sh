#!/usr/bin/env bash

#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="[M46|M40]"
#SBATCH --job-name=test_evops
#SBATCH --output=test/%j.out
#SBATCH --error=test/%j.err
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
lambda=${hyperparameters[2]}
match_mean=${hyperparameters[3]}
holdout=${hyperparameters[4]}

echo "TF: $tf"
echo "Target: $target"
echo "Lambda: $lambda"
echo "Match Mean: $match_mean"
echo "Holdout Index: $holdout"
printf "\n"

# Get ready to saved the log
ROOT=$1
TEST_LOGS="${ROOT}/logs/test"
mkdir -p $TEST_LOGS

# (1) Activate conda
echo 'activating conda environment'
source activate genomic_tools
which python

# (2) Now run a single test job
echo 'running script'
python test.py "$tf" "$target" "$lambda" "$match_mean" "$holdout" > "$TEST_LOGS/EvoPS_${tf}_${target}_${holdout}.log"