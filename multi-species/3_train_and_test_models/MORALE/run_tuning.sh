#!/usr/bin/env bash

set -e

ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/MORALE/multi-species"

# (0) Ensure the same types of directories we need exist

# Where all log files will be written to
# log files contain performance measurements
# made each epoch for the model.
log_root="$ROOT/logs"
mkdir -p "$log_root"

# (1) Set type of tuning we want to do

TUNE_TYPE="RANDOM" # RANDOM, GRID

# TFs to train models over. We have the following options:
## "CEBPA" "FOXA1" "HNF4A" "HNF6"
tfs=( "FOXA1" "HNF4A" "HNF6" )

# The target species to train the model on. We have the
# following options:
## "mm10" "rn7" "rheMac10" "canFam6" "hg38"
targets=( "mm10" "rn7" "rheMac10" "canFam6" )

# Set the hyperparameters to tune over
to_match=( 0 1 )
lambdas=()

# Populate the lambda hyperparameters
if [ "$TUNE_TYPE" == "GRID" ]; then
	for i in $(seq 0.0 0.5 3.0); do
		parameters+=( $(printf "%.2f" "$i") )
	done
elif [ "$TUNE_TYPE" == "RANDOM" ]; then
	for i in $(seq 1.0 1.0 8.0); do
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
	for target in "${targets[@]}"; do
		for _match_mean in "${to_match[@]}"; do
			for ((i =0; i < num_trials; i++)); do
				_lambda=${parameters[$i]}

				# Now write line to tune_array.txt
				printf "%s,%s,%s,%s\n" $tf $target $_match_mean $_lambda >> tune_array.txt
			done
		done
	done
done

total_runs=$(cat tune_array.txt | wc -l)

sbatch --array=1-${total_runs}%20 tune.sh $ROOT