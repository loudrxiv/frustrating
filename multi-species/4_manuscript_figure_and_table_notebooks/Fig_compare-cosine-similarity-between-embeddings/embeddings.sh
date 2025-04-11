#!/usr/bin/env bash

#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="[M46|M40]"
#SBATCH --job-name=embed
#SBATCH --output=embedding/%j.out
#SBATCH --error=embedding/%j.err
#SBATCH --mem=220G
date;hostname;id
printf "========================\n"

# Adjust to your desired line number
desired_line=$SLURM_ARRAY_TASK_ID

# Use sed to extract the specific line
line=$(sed -n "${desired_line}p" embedding_array.txt)

# Split the line into an array using ',' as a delimiter
IFS=',' read -r -a hyperparameters <<< "$line"

# Print the results
tf=${hyperparameters[0]}
target=${hyperparameters[1]}
lambda=${hyperparameters[2]}
match_mean=${hyperparameters[3]}

echo "TF: $tf"
echo "Target: $target"
echo "Lambda: $lambda"
echo "Match Mean: $match_mean"
printf "\n"

# Get ready to saved the log
ROOT=$1
EMBED_LOGS="${ROOT}/logs/embeddings"
mkdir -p $EMBED_LOGS

# (1) Activate conda
echo 'activating conda environment'
source activate genomic_tools
which python

# (2) Now run a single job
echo 'running script'
python embeddings.py "$tf" "$target" "$lambda" "$match_mean" > "$EMBED_LOGS/MORALE_${tf}_${target}.log"