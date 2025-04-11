#!/usr/bin/env bash

set -e

ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/MORALE/multi-species"
LOG_ROOT="$ROOT/logs"

# (0) Ensure the same types of directories we need exist

# Where all log files will be written to
# log files contain performance measurements
# made each epoch for the model.
mkdir -p "$LOG_ROOT"

# (1) Set type of tuning we want to do

TUNE_TYPE="RANDOM" # RANDOM, GRID
NUM_LAMBDAS=3

# TFs to train models over. We have the following options:
## "CEBPA" "FOXA1" "HNF4A" "HNF6"
tfs=( "HNF4A" "HNF6" )

# The target species to train the model on. We have the
# following options:
## "mm10" "rn7" "rheMac10" "canFam6" "hg38"
targets=( "mm10" "rn7" "rheMac10" "canFam6" )

# The index of the holdout to use for training
# We have the following options:
## 0 1 2 3
holdouts=( 0 1 2 3 )

# (2) Set the hyperparameters to tune over

should_match=( 0 1 )
lambdas=()
if [ "$TUNE_TYPE" == "GRID" ]; then
	for i in $(seq 0.0 0.5 3.0); do
		lambdas+=( $(printf "%.2f" "$i") )
	done
elif [ "$TUNE_TYPE" == "RANDOM" ]; then
	for i in $(seq 1.0 1.0 8.0); do
		lambdas+=( $(printf "%.2f" "$i") )
	done

	# Shuffle the lambdas and grab NUM_LAMBDAS amount
	shuf_lambdas=($(shuf -e "${lambdas[@]}"))
	lambdas=("${shuf_lambdas[@]:0:$NUM_LAMBDAS}")

else
	echo "Invalid TUNE_TYPE: $TUNE_TYPE"
fi

num_lambdas=${#lambdas[@]}

if [ -f tune_array.txt ]; then
	rm tune_array.txt
	touch tune_array.txt
else 
	touch tune_array.txt
fi

# We will train a model for each TF on each target species
for tf in "${tfs[@]}"; do
	for target in "${targets[@]}"; do
		for holdout in "${holdouts[@]}"; do
			for match_mean in "${should_match[@]}"; do
				for ((i =0; i < num_lambdas; i++)); do
					_lambda=${lambdas[$i]}
					printf "%s,%s,%s,%s,%s\n" $tf $target $_lambda $match_mean $holdout >> tune_array.txt
				done
			done
		done
	done
done

total_runs=$(cat tune_array.txt | wc -l)

sbatch --array=1-${total_runs}%20 tune.sh $ROOT