#!/usr/bin/env bash

set -e

ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/MORALE/multi-species"
LOG_ROOT="$ROOT/logs"

# (0) Ensure the same types of directories we need exist

# Where all log files will be written to
# log files contain performance measurements
# made each epoch for the model.
mkdir -p "$LOG_ROOT"

# The python script below is also expecting a save directory for 
# models to exist
models_dir="$ROOT/models"
mkdir -p "$models_dir"

# TFs to train models over. We have the following options:
## "CEBPA" "FOXA1" "HNF4A" "HNF6"
tfs=( "CEBPA" "FOXA1" "HNF4A" "HNF6" )

# The index of the num_holdout to use for training
# We have the following options:
## 0 1 2 3
num_holdouts=( 0 1 2 3 )

# Create the train_array.txt file

if [ -f train_array.txt ]; then
	rm train_array.txt
	touch train_array.txt
fi

target="hg38" # always do hg38 here
for tf in "${tfs[@]}"; do
	for num_holdout in "${num_holdouts[@]}"; do

		if [ "$tf" == "CEBPA" ]; then
			if [ "$num_holdout" == "0" ]; then
				_lambda="3.0"
				_match_mean="1"
			elif [ "$num_holdout" == "1" ]; then
				_lambda="3.0"
				_match_mean="1"
			elif [ "$num_holdout" == "2" ]; then
				_lambda="8.0"
				_match_mean="1"
			elif [ "$num_holdout" == "3" ]; then
				_lambda="8.0"
				_match_mean="1"
			fi
		elif [ "$tf" == "FOXA1" ]; then
			if [ "$num_holdout" == "0" ]; then
				_lambda="8.0"
				_match_mean="1"
			elif [ "$num_holdout" == "1" ]; then
				_lambda="5.0"
				_match_mean="1"
			elif [ "$num_holdout" == "2" ]; then
				_lambda="7.0"
				_match_mean="1"
			elif [ "$num_holdout" == "3" ]; then
				_lambda="3.0"
				_match_mean="1"
			fi
		elif [ "$tf" == "HNF4A" ]; then
			if [ "$num_holdout" == "0" ]; then
				_lambda="8.0"
				_match_mean="0"
			elif [ "$num_holdout" == "1" ]; then
				_lambda="8.0"
				_match_mean="1"
			elif [ "$num_holdout" == "2" ]; then
				_lambda="8.0"
				_match_mean="1"
			elif [ "$num_holdout" == "3" ]; then
				_lambda="7.0"
				_match_mean="1"
			fi
		elif [ "$tf" == "HNF6" ]; then
			if [ "$num_holdout" == "0" ]; then
				_lambda="6.0"
				_match_mean="0"
			elif [ "$num_holdout" == "1" ]; then
				_lambda="4.0"
				_match_mean="1"
			elif [ "$num_holdout" == "2" ]; then
				_lambda="5.0"
				_match_mean="1"
			elif [ "$num_holdout" == "3" ]; then
				_lambda="3.0"
				_match_mean="1"
			fi
		else
			printf "Invalid TF: $tf\n"
			exit 1
		fi

		printf "%s,%s,%s,%s,%s\n" $tf $target $_lambda $_match_mean $num_holdout >> train_array.txt
	done
done

total_runs=$(cat train_array.txt | wc -l)

sbatch --array=1-${total_runs} train.sh $ROOT