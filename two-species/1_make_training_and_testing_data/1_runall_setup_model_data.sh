#!/usr/bin/env bash

#-------------------------------------------------------------------------------
# This script will run all the adjacent scripts to set up datasets for model
# training and testing. See inside individual scripts for their purpose.

# NOTE: These scripts have specific directory structure expectations, and in
# particular, they require the "all.all" files created by earlier scripts.
#-------------------------------------------------------------------------------

set -e

tfs=( "CTCF" "CEBPA" "HNF4A" "RXRA" )
genomes=( "mm10" "hg38" )

ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/two-species/tensorflow"

# One script is a special exception -- we need to create some repeat files.
# The script needs an all.all file (produced by previous steps) to do it. 
# So we can pick any TF's file to pass in.
for genome in "${genomes[@]}"; do
	random_tf="CTCF"  # this can be any TF you've got an all.all file for
	allall_file="$ROOT/raw_data/${genome}/${random_tf}/all.all"
	./1.0_make_repeat_files.sh "$ROOT" "$allall_file" "$genome"  || exit 1
done

for tf in "${tfs[@]}"; do
	for genome in "${genomes[@]}"; do
		echo "Setting up training data for ${tf} + ${genome}..."
		./1.1_make_val_test_files_and_prep_training_files.sh "$ROOT" "$tf" "$genome"  || exit 1
		./1.2_make_neg_window_files_for_epochs.sh "$ROOT" "$tf" "$genome"  || exit 1
	done

	# This script loops over the genomes internally,
    # because it needs to look at data from both at the same time
	./1.3_make_species_files_for_epochs.sh "$ROOT" "$tf"  || exit 1

done