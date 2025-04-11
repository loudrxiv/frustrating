#!/usr/bin/env bash

set -e

ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/two-species/tensorflow"

# Where all log files will be written to
# log files contain performance measurements
# made each epoch for the model.
log_root="$ROOT/logs"
mkdir -p "$log_root"

# The python script below is also expecting a save directory for 
# models to exist
models_dir="$ROOT/models"
mkdir -p "$models_dir"

# TFs to train models over. We have the following options:
## "CTCF" "CEBPA" "HNF4A" "RXRA"
tfs=( "HNF4A" "RXRA" "CTCF" "CEBPA" ) 

# The target species to train the model on. We have the
# following options:
## "mm10" "hg38"
sources=( "mm10" "hg38" )

# The run number for the model. This is used to because we 
# implement the previous 5-fold cross validation
# scheme
runs=( 1 2 3 4 5 )

if [ -f train_array.txt ]; then
	rm train_array.txt
	touch train_array.txt
fi

# We will train a model for each TF on each target species in a 
# SLURM array
for tf in "${tfs[@]}"; do
	for source in "${sources[@]}"; do
		for run in "${runs[@]}"; do
			printf "%s,%s,%s\n" $tf $source $run >> train_array.txt
		done
	done
done

total_runs=$(cat train_array.txt | wc -l)

sbatch --array=1-${total_runs}%20 train.sh $ROOT $log_root