#!/usr/bin/env bash

set -e

ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/two-species/tensorflow"

# (0) Ensure the same types of directories we need exist

# Where all log files will be written to
# log files contain performance measurements
# made each epoch for the model.
log_root="$ROOT/logs"
mkdir -p "$log_root"

# (1) Set type of tuning we want to do

TUNE_TYPE="RANDOM" # RANDOM, GRID

# (2) Set the other parameters for the tuning runs

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
runs=( 1 ) # we just do it on a single run

# Set the lambas to tune over
parameters=()

if [ "$TUNE_TYPE" == "GRID" ]; then
	for i in $(seq 0.0 0.5 3.0); do
		parameters+=( $(printf "%.2f" "$i") )
	done
	#num_trials=${#parameters[@]}
elif [ "$TUNE_TYPE" == "RANDOM" ]; then
	#num_trials=25
	#for i in $(seq 0.0 0.01 10.0 | shuf); do
	for i in $(seq 0.0 0.5 10.0); do
		parameters+=( $(printf "%.2f" "$i") )
	done
else
	echo "Invalid TUNE_TYPE: $TUNE_TYPE"
fi

num_trials=${#parameters[@]}

if [ -f tune_array.txt ]; then
	rm tune_array.txt
	touch tune_array.txt
else 
	touch tune_array.txt
fi

# We will train a model for each TF on each target species
for tf in "${tfs[@]}"; do
	for target in "${sources[@]}"; do
        for run in "${runs[@]}"; do
            for ((i =0; i < num_trials; i++)); do

                lambda_=${parameters[$i]}

                # Now write line to tune_array.txt
                printf "%s,%s,%s,%s\n" $tf $target $run $lambda_ >> tune_array.txt
            done
		done
	done
done

total_runs=$(cat tune_array.txt | wc -l)

sbatch --array=1-${total_runs}%20 tune.sh $ROOT