#!/usr/bin/env bash

#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="[M46|M40|M24|M20]"
#SBATCH --job-name=tuning_grl
#SBATCH --output=tune/%j.out
#SBATCH --error=tune/%j.err
#SBATCH --exclude=g023
#SBATCH --mem=180G
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
source=${hyperparameters[1]}
run=${hyperparameters[2]}
lambda_=${hyperparameters[3]}

echo "TF: $tf"
echo "Source: $source"
echo "Run: $run"
echo "Lambda: $lambda_"
printf "\n"

# Get ready to saved the log
ROOT=$1
TUNE_LOGS="${ROOT}/logs/tune"
mkdir -p $TUNE_LOGS

# (1) Activate conda
echo 'activating conda environment'
source activate tensorflow_A2
which python

# (2) Point to the tuning file we need
tune_file=${ROOT}/src/2_train_and_test_models/GRL/tune.py
echo 'tune file: ' ${tune_file}

# (3) Now run a single tuning job 
echo 'running script'
python tune.py "$tf" "$source" "$run" "$lambda_" > "$TUNE_LOGS/GRL_tf-${tf}_source-${source}_run-${run}_lamb-${lambda_}.log"