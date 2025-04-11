#!/usr/bin/env bash

set -e

ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/MORALE/multi-species"

# The python script below is also expecting a save directory for 
# preds and labels to exist
out_dir="$ROOT/model_out"
mkdir -p "$out_dir"

# TFs to train models over. We have the following options:
## "CEBPA" "FOXA1" "HNF4A" "HNF6"
tfs=( "CEBPA" "FOXA1" "HNF4A" "HNF6" )

# The number of 'closely related' species to hold out
# during training.
## 0 1 2 3
holdouts=( 0 1 2 3 )

if [ -f test_array.txt ]; then
	rm test_array.txt
	touch test_array.txt
fi

target="hg38" # always do hg38 here
for tf in "${tfs[@]}"; do
	for holdout in "${holdouts[@]}"; do

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

		printf "%s,%s,%s,%s,%s\n" $tf $target $_lambda $_match_mean $holdout >> test_array.txt

	done
done

total_runs=$(cat test_array.txt | wc -l)

sbatch --array=1-${total_runs}%20 test.sh $ROOT