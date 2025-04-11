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
tfs=( "CTCF" "CEBPA" "HNF4A" "RXRA" ) 

# The source species to train the model on. We have the
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

# We will train a model for each TF
for tf in "${tfs[@]}"; do
	for source in "${sources[@]}"; do
		for run in "${runs[@]}"; do
		
			if [ "$source" == "hg38" ]; then
				if [ "$tf" == "CEBPA" ]; then
					lambda_="7.0"
					match="1"
					# 0.2797990475578469, 0.2875005810345951
				elif [ "$tf" == "CTCF" ]; then
					lambda_="4.0"
					match="1"
					# 0.6102941138477869, 0.6990186674550396
				elif [ "$tf" == "HNF4A" ]; then
					lambda_="8.0"
					match="1"
					# 0.24424882915776389, 0.2764845806329447
				elif [ "$tf" == "RXRA" ]; then
					lambda_="8.0"
					match="0"
					# 0.21038397084229232, 0.3263132161625715
				else
					printf "Invalid TF: $tf\n"
					exit 1
				fi
			elif [ "$source" == "mm10" ]; then
				if [ "$tf" == "CEBPA" ]; then
					lambda_="8.0"
					match="0"
					# 0.24863538855972514, 0.36616049155507213
				elif [ "$tf" == "CTCF" ]; then
					lambda_="4.0"
					match="0"
					# 0.6458876708489377, 0.7069821656079107
				elif [ "$tf" == "HNF4A" ]; then
					lambda_="6.0"
					match="0"
					# 0.21674723512185973, 0.28873938750755773
				elif [ "$tf" == "RXRA" ]; then
					lambda_="7.0"
					match="0"
					# 0.20542190826940987, 0.24081767937478246
				else
					printf "Invalid TF: $tf\n"
					exit 1
				fi
			else 
				printf "Invalid source: $source\n"
				exit 1
			fi

			# Now write line to train_array.txt
			printf "%s,%s,%s,%s,%s\n" $tf $source $run $match $lambda_ >> train_array.txt

		done
	done
done

total_runs=$(cat train_array.txt | wc -l)

sbatch --array=1-${total_runs}%20 train.sh $ROOT