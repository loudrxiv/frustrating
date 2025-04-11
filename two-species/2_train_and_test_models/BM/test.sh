#!/usr/bin/env bash

#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="[M46|M40|M24|M20|M16]"
#SBATCH --job-name=testing_bm
#SBATCH --output=test/%j.out
#SBATCH --error=test/%j.err
#SBATCH --exclude=g023
#SBATCH --mem=180G
date;hostname;id
printf "========================\n"

export TF_CPP_MIN_LOG_LEVEL=1

# Adjust to your desired line number
desired_line=$SLURM_ARRAY_TASK_ID

# Use sed to extract the specific line
line=$(sed -n "${desired_line}p" test_array.txt)

# Split the line into an array using ',' as a delimiter
IFS=',' read -r -a hyperparameters <<< "$line"

# Print the results
tf=${hyperparameters[0]}
source=${hyperparameters[1]}
domain=${hyperparameters[2]}

echo "TF: $tf"
echo "Source: $source"
echo "Domain: $domain"
printf "\n"

# Get ready to save the log
ROOT=$1
LOG_ROOT=$2

TEST_LOGS="${LOG_ROOT}/test"
mkdir -p "$TEST_LOGS"

# (1) Activate conda
echo 'activating conda environment'
source activate tensorflow_A2
which python

# (2) Point to the testing file we need
test_file=${ROOT}/2_train_and_test_models/BM/test.py
echo 'test file: ' ${test_file}

# (3) Now run a single testing job 
echo 'running script'
python test.py "$tf" "$source" "$domain" > "$TEST_LOGS/BM_tf-${tf}_trained-${source}_tested-${domain}.log"