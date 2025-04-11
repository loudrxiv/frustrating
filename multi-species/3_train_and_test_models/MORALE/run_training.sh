#!/usr/bin/env bash

set -e

ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/MORALE/multi-species"
LOG_ROOT="$ROOT/logs"

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

# The target species to train the model on. We have the
# following options:
## "mm10" "rn7" "rheMac10" "canFam6" "hg38"
targets=( "mm10" "rn7" "rheMac10" "canFam6" )

if [ -f train_array.txt ]; then
	rm train_array.txt
	touch train_array.txt
fi

# We will train a model for each TF on each target species
for tf in "${tfs[@]}"; do
	for target in "${targets[@]}"; do

		if [ "$target" == "hg38" ]; then
			if [ "$tf" == "CEBPA" ]; then
				_lambda="7.0"
				_match_mean="1"
			elif [ "$tf" == "FOXA1" ]; then
				_lambda="5.0"
				_match_mean="1"
			elif [ "$tf" == "HNF4A" ]; then
				_lambda="8.0"
				_match_mean="0"
			elif [ "$tf" == "HNF6" ]; then
				_lambda="3.0"
				_match_mean="1"
			else
				printf "Invalid TF: $tf\n"
				exit 1
			fi
		elif [ "$target" == "mm10" ]; then
			if [ "$tf" == "CEBPA" ]; then
				_lambda="7.0"
				_match_mean="1"
			elif [ "$tf" == "FOXA1" ]; then
				_lambda="5.0"
				_match_mean="1"
			elif [ "$tf" == "HNF4A" ]; then
				_lambda="3.0"
				_match_mean="0"
			elif [ "$tf" == "HNF6" ]; then
				_lambda="7.0"
				_match_mean="1"
			else
				printf "Invalid TF: $tf\n"
				exit 1
			fi
		elif [ "$target" == "rn7" ]; then
			if [ "$tf" == "CEBPA" ]; then
				_lambda="2.0"
				_match_mean="0"
			elif [ "$tf" == "FOXA1" ]; then
				_lambda="3.0"
				_match_mean="1"
			elif [ "$tf" == "HNF4A" ]; then
				_lambda="8.0"
				_match_mean="0"
			elif [ "$tf" == "HNF6" ]; then
				_lambda="5.0"
				_match_mean="1"
			else
				printf "Invalid TF: $tf\n"
				exit 1
			fi
		elif [ "$target" == "rheMac10" ]; then
			if [ "$tf" == "CEBPA" ]; then
				_lambda="1.0"
				_match_mean="0"
			elif [ "$tf" == "FOXA1" ]; then
				_lambda="6.0"
				_match_mean="1"
			elif [ "$tf" == "HNF4A" ]; then
				_lambda="6.0"
				_match_mean="1"
			elif [ "$tf" == "HNF6" ]; then
				_lambda="4.0"
				_match_mean="0"
			else
				printf "Invalid TF: $tf\n"
				exit 1
			fi
		elif [ "$target" == "canFam6" ]; then
			if [ "$tf" == "CEBPA" ]; then
				_lambda="8.0"
				_match_mean="0"
			elif [ "$tf" == "FOXA1" ]; then
				_lambda="4.0"
				_match_mean="1"
			elif [ "$tf" == "HNF4A" ]; then
				_lambda="5.0"
				_match_mean="0"
			elif [ "$tf" == "HNF6" ]; then
				_lambda="5.0"
				_match_mean="0"
			else
				printf "Invalid TF: $tf\n"
				exit 1
			fi
		fi

		# Now write line to train_array.txt
		printf "%s,%s,%s,%s\n" $tf $target $_lambda $_match_mean >> train_array.txt

	done
done

total_runs=$(cat train_array.txt | wc -l)

sbatch --array=1-${total_runs} train.sh $ROOT