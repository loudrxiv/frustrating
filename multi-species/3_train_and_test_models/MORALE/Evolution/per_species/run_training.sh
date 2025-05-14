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
tfs=( "FOXA1" )

# Target species to predict on, we have the following options:
## "hg38" "mm10" "rn7" "canFam6" "rheMac10"
targets=( "rheMac10" )

# The index of the holdout to use for training
# We have the following options:
## 0 1 2 3
holdouts=( 3 )

# Create the train_array.txt file

if [ -f train_array.txt ]; then
	rm train_array.txt
	touch train_array.txt
fi

for target in "${targets[@]}"; do
	for tf in "${tfs[@]}"; do
		for holdout in "${holdouts[@]}"; do

			if [ "$target" == "hg38" ]; then
				if [ "$tf" == "CEBPA" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="4.0"
						_match_mean="1"
					elif [ "$holdout" == "1" ]; then
						_lambda="5.0"
						_match_mean="1"
					elif [ "$holdout" == "2" ]; then
						_lambda="1.0"
						_match_mean="0"
					elif [ "$holdout" == "3" ]; then
						_lambda="4.0"
						_match_mean="0"
					fi
				elif [ "$tf" == "FOXA1" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="4.0"
						_match_mean="1"
					elif [ "$holdout" == "1" ]; then
						_lambda="6.0"
						_match_mean="1"
					elif [ "$holdout" == "2" ]; then
						_lambda="6.0"
						_match_mean="1"
					elif [ "$holdout" == "3" ]; then
						_lambda="6.0"
						_match_mean="1"
					fi
				elif [ "$tf" == "HNF4A" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="3.0"
						_match_mean="1"
					elif [ "$holdout" == "1" ]; then
						_lambda="3.0"
						_match_mean="1"
					elif [ "$holdout" == "2" ]; then
						_lambda="4.0"
						_match_mean="1"
					elif [ "$holdout" == "3" ]; then
						_lambda="3.0"
						_match_mean="1"
					fi
				elif [ "$tf" == "HNF6" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="4.0"
						_match_mean="1"
					elif [ "$holdout" == "1" ]; then
						_lambda="6.0"
						_match_mean="1"
					elif [ "$holdout" == "2" ]; then
						_lambda="4.0"
						_match_mean="0"
					elif [ "$holdout" == "3" ]; then
						_lambda="6.0"
						_match_mean="0"
					fi
				else
					printf "Invalid TF: $tf\n"
					exit 1
				fi
			elif [ "$target" == "mm10" ]; then
				if [ "$tf" == "CEBPA" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="4.0"
						_match_mean="0"
					elif [ "$holdout" == "1" ]; then
						_lambda="7.0"
						_match_mean="1"
					elif [ "$holdout" == "2" ]; then
						_lambda="8.0"
						_match_mean="0"
					elif [ "$holdout" == "3" ]; then
						_lambda="7.0"
						_match_mean="1"
					fi
				elif [ "$tf" == "FOXA1" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="7.0"
						_match_mean="0"
					elif [ "$holdout" == "1" ]; then
						_lambda="4.0"
						_match_mean="0"
					elif [ "$holdout" == "2" ]; then
						_lambda="8.0"
						_match_mean="0"
					elif [ "$holdout" == "3" ]; then
						_lambda="4.0"
						_match_mean="1"
					fi
				elif [ "$tf" == "HNF4A" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="6.0"
						_match_mean="0"
					elif [ "$holdout" == "1" ]; then
						_lambda="7.0"
						_match_mean="0"
					elif [ "$holdout" == "2" ]; then
						_lambda="2.0"
						_match_mean="0"
					elif [ "$holdout" == "3" ]; then
						_lambda="2.0"
						_match_mean="1"
					fi
				elif [ "$tf" == "HNF6" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="1.0"
						_match_mean="1"
					elif [ "$holdout" == "1" ]; then
						_lambda="8.0"
						_match_mean="1"
					elif [ "$holdout" == "2" ]; then
						_lambda="3.0"
						_match_mean="1"
					elif [ "$holdout" == "3" ]; then
						_lambda="3.0"
						_match_mean="1"
					fi
				else
					printf "Invalid TF: $tf\n"
					exit 1
				fi
			elif [ "$target" == "rn7" ]; then
				if [ "$tf" == "CEBPA" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="4.0"
						_match_mean="0"
					elif [ "$holdout" == "1" ]; then
						_lambda="7.0"
						_match_mean="1"
					elif [ "$holdout" == "2" ]; then
						_lambda="8.0"
						_match_mean="0"
					elif [ "$holdout" == "3" ]; then
						_lambda="7.0"
						_match_mean="0"
					fi
				elif [ "$tf" == "FOXA1" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="4.0"
						_match_mean="0"
					elif [ "$holdout" == "1" ]; then
						_lambda="4.0"
						_match_mean="0"
					elif [ "$holdout" == "2" ]; then
						_lambda="7.0"
						_match_mean="1"
					elif [ "$holdout" == "3" ]; then
						_lambda="7.0"
						_match_mean="1"
					fi
				elif [ "$tf" == "HNF4A" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="7.0"
						_match_mean="1"
					elif [ "$holdout" == "1" ]; then
						_lambda="2.0"
						_match_mean="0"
					elif [ "$holdout" == "2" ]; then
						_lambda="2.0"
						_match_mean="0"
					elif [ "$holdout" == "3" ]; then
						_lambda="7.0"
						_match_mean="1"
					fi
				elif [ "$tf" == "HNF6" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="6.0"
						_match_mean="1"
					elif [ "$holdout" == "1" ]; then
						_lambda="6.0"
						_match_mean="0"
					elif [ "$holdout" == "2" ]; then
						_lambda="2.0"
						_match_mean="1"
					elif [ "$holdout" == "3" ]; then
						_lambda="6.0"
						_match_mean="1"
					fi
				else
					printf "Invalid TF: $tf\n"
					exit 1
				fi
			elif [ "$target" == "canFam6" ]; then
				if [ "$tf" == "CEBPA" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="7.0"
						_match_mean="0"
					elif [ "$holdout" == "1" ]; then
						_lambda="4.0"
						_match_mean="0"
					elif [ "$holdout" == "2" ]; then
						_lambda="4.0"
						_match_mean="1"
					elif [ "$holdout" == "3" ]; then
						_lambda="7.0"
						_match_mean="0"
					fi
				elif [ "$tf" == "FOXA1" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="8.0"
						_match_mean="0"
					elif [ "$holdout" == "1" ]; then
						_lambda="8.0"
						_match_mean="1"
					elif [ "$holdout" == "2" ]; then
						_lambda="7.0"
						_match_mean="0"
					elif [ "$holdout" == "3" ]; then
						_lambda="8.0"
						_match_mean="0"
					fi
				elif [ "$tf" == "HNF4A" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="2.0"
						_match_mean="1"
					elif [ "$holdout" == "1" ]; then
						_lambda="7.0"
						_match_mean="0"
					elif [ "$holdout" == "2" ]; then
						_lambda="2.0"
						_match_mean="1"
					elif [ "$holdout" == "3" ]; then
						_lambda="2.0"
						_match_mean="0"
					fi
				elif [ "$tf" == "HNF6" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="6.0"
						_match_mean="0"
					elif [ "$holdout" == "1" ]; then
						_lambda="7.0"
						_match_mean="0"
					elif [ "$holdout" == "2" ]; then
						_lambda="6.0"
						_match_mean="0"
					elif [ "$holdout" == "3" ]; then
						_lambda="2.0"
						_match_mean="1"
					fi
				else
					printf "Invalid TF: $tf\n"
					exit 1
				fi
			elif [ "$target" == "rheMac10" ]; then
				if [ "$tf" == "CEBPA" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="8.0"
						_match_mean="1"
					elif [ "$holdout" == "1" ]; then
						_lambda="4.0"
						_match_mean="1"
					elif [ "$holdout" == "2" ]; then
						_lambda="4.0"
						_match_mean="1"
					elif [ "$holdout" == "3" ]; then
						_lambda="4.0"
						_match_mean="1"
					fi
				elif [ "$tf" == "FOXA1" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="8.0"
						_match_mean="1"
					elif [ "$holdout" == "1" ]; then
						_lambda="8.0"
						_match_mean="1"
					elif [ "$holdout" == "2" ]; then
						_lambda="8.0"
						_match_mean="1"
					elif [ "$holdout" == "3" ]; then
						_lambda="4.0"
						_match_mean="0"
					fi
				elif [ "$tf" == "HNF4A" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="6.0"
						_match_mean="1"
					elif [ "$holdout" == "1" ]; then
						_lambda="7.0"
						_match_mean="0"
					elif [ "$holdout" == "2" ]; then
						_lambda="2.0"
						_match_mean="0"
					elif [ "$holdout" == "3" ]; then
						_lambda="6.0"
						_match_mean="0"
					fi
				elif [ "$tf" == "HNF6" ]; then
					if [ "$holdout" == "0" ]; then
						_lambda="7.0"
						_match_mean="1"
					elif [ "$holdout" == "1" ]; then
						_lambda="7.0"
						_match_mean="0"
					elif [ "$holdout" == "2" ]; then
						_lambda="2.0"
						_match_mean="1"
					elif [ "$holdout" == "3" ]; then
						_lambda="6.0"
						_match_mean="1"
					fi
				else
					printf "Invalid TF: $tf\n"
					exit 1
				fi
			else
				printf "Invalid target: $target\n"
				exit 1
			fi

			printf "%s,%s,%s,%s,%s\n" $tf $target $_lambda $_match_mean $holdout >> train_array.txt

		done
	done
done

total_runs=$(cat train_array.txt | wc -l)

sbatch --array=1-${total_runs} train.sh $ROOT