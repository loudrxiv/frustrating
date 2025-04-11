#!/usr/bin/env bash

#SBATCH --partition=benos,dept_cpu,dept_gpu
#SBATCH --job-name=%x_%j
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

# Whereas previously, we manually downloaded each of the references, blacklists & etc. Now 
# we simply rely on genomepy (a nice package that bundles a ton of steps together for us!).
# Here is a link to the project:
# https://github.com/vanheeringen-lab/genomepy
#---------------------------------------------------------------------------------------------

RAW_DATA_DIR=$1
genome=$2

# We choose to always activate `genomic_tools`
source activate genomic_tools

# (2) Run genompy
# Because we have the blacklist and bowtie2 plugins enabled for genomepy,
# this will do all of the gathering of these resources for us.
genomepy install "$genome" -g "$RAW_DATA_DIR" --annotation