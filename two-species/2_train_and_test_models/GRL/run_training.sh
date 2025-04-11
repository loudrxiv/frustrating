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

			# Based on our tuned hyperparameters!G
			if [ "$source" == "hg38" ]; then
				if [ "$tf" == "CEBPA" ]; then
					lambda_="6.0"
					# 1.0: 0.26111348246320076, 0.25261339150641593
					# 6.00: 0.2658239584098164, 0.26698076458017905
				elif [ "$tf" == "CTCF" ]; then
					lambda_="1.5"
					# 1.00: 0.4907932250556362, 0.6766735033127771
					# 1.50: 0.5750271620936995, 0.6864416275044679
				elif [ "$tf" == "HNF4A" ]; then
					lambda_="0.5"
					# 1.00: 0.20619445697403416, 0.2505672899570722
					# 0.50: 0.21233410380111173, 0.25642501355363584
				elif [ "$tf" == "RXRA" ]; then
					lambda_="7.5"
					# 1.00: 0.20055820398867308, 0.2829772502429648
					# 7.50: 0.18864245847018726, 0.30155441406135347
				else
					printf "Invalid TF: $tf\n"
					exit 1
				fi
			elif [ "$source" == "mm10" ]; then
				if [ "$tf" == "CEBPA" ]; then
					lambda_="8.5"
					# 1.00: 0.24537712576277415, 0.3338379064725669
					# 8.50: 0.2343777206413144, 0.3418619276648583
				elif [ "$tf" == "CTCF" ]; then
					lambda_="6.5"
					# 1.00: 0.596701606269403, 0.6628469146834961
					# 6.50: 0.6217684381075879, 0.6882030187612246
				elif [ "$tf" == "HNF4A" ]; then
					lambda_="10.0"
					# 1.00: 0.20772998279364996, 0.2632310284808414
					# 10.00: 0.1928444527995804, 0.2782918003971279
				elif [ "$tf" == "RXRA" ]; then
					lambda_="1.0"
					# 1.0: 0.22494260394756618, 0.22688678621097885
					# 1.00: 0.2147760526129844, 0.23318797045528905
				else
					printf "Invalid TF: $tf\n"
					exit 1
				fi
			else 
				printf "Invalid source: $source\n"
				exit 1
			fi

			# Now write line to train_array.txt
			printf "%s,%s,%s,%s\n" $tf $source $run $lambda_ >> train_array.txt

		done
	done
done

total_runs=$(cat train_array.txt | wc -l)

sbatch --array=1-${total_runs}%20 train.sh $ROOT $log_root