#!/usr/bin/env bash

set -e

# NOTE: These scripts have specific directory structure expectations, and in
# particular, they require the "all.all" files created by earlier scripts.
#---------------------------------------------------------------------------------------------------

ROOT="$(pwd)/.."
TFS=( "CEBPA" "FOXA1" "HNF4A" "HNF6" )
GENOMES=( "mm10" "hg38" "rheMac10" "canFam6" "rn7" )

for genome in "${GENOMES[@]}"; do
	# We instantiate the genome-specific rmsk files
	DATA_ROOT="$ROOT/data/$genome"
	RMSK_FILE="$DATA_ROOT/rmsk.bed"

	#-- (1) List the repeat types and SINE subfamilies
	# The rmsk.bed files should of been downloaded when you setup the directories,
	# we confirm that here and proceed if everything is okay.
	if [[ ! -f "$DATA_ROOT/rmsk.bed" ]]; then
		printf "RepeatMasker track needs to be downloaded from UCSC."
		exit 1
	fi

	printf "Here are the repeat types and their counts for ${genome}:\n"
	./_count_repeat_types.sh "$DATA_ROOT/rmsk.bed"

	printf "Here are the subfamilies of the SINEs in ${genome}:\n"
	./_list_subfamilies.sh "$DATA_ROOT/rmsk.bed"

	# We create a dummy for the repeat elements annotations
	allall_file="$ROOT/raw_data/${genome}/CEBPA/all.all"
	./2-0_make_repeat_files.sh $ROOT $genome "$allall_file"

	for tf in "${TFS[@]}"; do
	
		printf "\n=== Setting up training data for ${tf} + ${genome}... ===\n\n"

		# We create files for the positive and negative windows
		./2-1_make_val_test_files_and_prep_training_files.sh "$ROOT" "$tf" "$genome" || exit 1

		# We create files for the species
		./2-2_make_species_files.sh "$ROOT" "$tf" "$genome"  || exit 1

	done

done