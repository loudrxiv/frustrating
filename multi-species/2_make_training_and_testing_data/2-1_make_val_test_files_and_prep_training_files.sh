#!/usr/bin/env bash

set -e

# This script begins the process of making all the files that
# models will use for training, validation, and testing. Specifically,
# this script creates the validation and testing set files for a
# given TF and species, creates the binding training set's bound example files,
# and preps the files that will be used to create the rest of the
# training data: the binding training set's unbound example files
# and the species-background training data files.
#
# NOTE: You NEED the all.all file create from previous scripts to be able to 
# generate our data.
#
# The all.all should be a file of all windows to be used for training/testing
# of models from a genome. This file is specific to each TF because it contains
# a column of binary binding labels (TF bound = 1, else 0). The format is TSV,
# with col 1-3 containing BED-format chromosome, start, stop info,
# and the final column containing binary binding labels for a given TF.
# This file is created by genomic window filtering scripts, so it does not
# contain regions filtered (hg38/mm10 ENCODE blacklist for example).
#---------------------------------------------------------------------------------------------------

#--- (1) Setup for the chromosome holdout removal!!

source activate causal_v240416 # we need R

ROOT=$1		# the directory for the project (same across all scripts)
tf=$2		# one of CTCF, CEBPA, Hnf4a, or RXRA
genome=$3	# one of mm10, hg38

echo "Prepping training datasets for $tf ($genome)."

RAW_DATA_DIR="$ROOT/raw_data/$genome/$tf"
DATA_DIR="$ROOT/data/$genome/$tf"
DATA_ROOT="$ROOT/data/$genome"

# Other files we WILL need/make
allfile="$RAW_DATA_DIR/all.all"
output_file="$DATA_DIR/chromosome_stats.tsv"

val_file="$DATA_DIR/val.bed"
val_shuf_file="$DATA_DIR/val_shuf.bed"

test_file="$DATA_DIR/test.bed"
test_shuf_file="$DATA_DIR/test_shuf.bed"

if [[ ! -f "$allfile" ]]; then
	echo "File all.all is missing from $RAW_DATA_DIR. Exiting."
	exit 1
fi

allbed=$allfile

total_windows=`wc -l < "$allbed"`
bound_windows=`awk '$NF == 1' "$allbed" | wc -l`
frac_bound=$(bc <<< "scale=4; $bound_windows/$total_windows")

echo "Total windows: $total_windows, of which $bound_windows are bound ($frac_bound)."

#--- (2) Making out statistics file!

# Create output file for the linear program
echo -e "Chromosome\tPositive\tNegative" > "$output_file"

# We create our holdouts at once: first the validation, now the test
# and they can span more than 1 chromosome! We opt to have a validation 
# set that is 10% of the data, balanced. As well as a test set that is 
# 10% of the data, balanced.
chrs_left=$(cut -f1 $allfile | sort | uniq | wc -l)

for i in $(seq 1 ${chrs_left}); do

	# Set up rations for the weights of the bound and unbound fractions
	total_chr=$(grep -F "chr${i}	" "$allbed" | wc -l)
	bound_chr=$(grep -F "chr${i}	" "$allbed" | awk '$NF == 1' | wc -l)
	unbound_chr=$(grep -F "chr${i}	" "$allbed" | awk '$NF == 0' | wc -l)

    # Append to the output file for the current chromosome (tab-separated)
    echo -e "chr${i}\t$bound_chr\t$unbound_chr" >> "$output_file"

done

#--- (3) Making the validation and test files (we go sequentially)!

# Call the R script and capture the output. We don't pass in
# the chromosome % we want as we just need to hardcode that 
# we need two out of the 19 chromosomes for validation and testing.
val_result=$(Rscript ${ROOT}/2_make_training_and_testing_data/_solve_class_balance.R $output_file)
IFS=',' read -r -a val_chrs <<< "${val_result//[\[\]]/}"

# Then run for the test set
test_result=$(Rscript ${ROOT}/2_make_training_and_testing_data/_solve_class_balance.R $output_file)
IFS=',' read -r -a test_chrs <<< "${test_result//[\[\]]/}"

# Grab chrs for validation and testing

# Validation
stripped_val_chrs=()
for chr in "${val_chrs[@]}"; do
  stripped_chr="${chr//\"/}"  # Remove all double quotes
  grep -F -w "$stripped_chr" "$allbed" | shuf >> "$val_file"
  stripped_val_chrs+=("$stripped_chr")
done

# Create shuffled version of the validation set
shuf "$val_file" > "$val_shuf_file"

val_windows=`wc -l < "$val_file" `
echo "Val set windows: $val_windows"

# Testing
stripped_test_chrs=()
for chr in "${test_chrs[@]}"; do
  stripped_chr="${chr//\"/}"  # Remove all double quotes
  grep -F -w "$stripped_chr" "$allbed" | shuf >> "$test_file"
  stripped_test_chrs+=("$stripped_chr")
done

# Create shuffled version of the test set
shuf "$test_file" > "$test_shuf_file"

test_windows=`wc -l < "$test_file" `
echo "Test set windows: $test_windows"

#--- Get training chromosomes, split into bound/unbound examples
# Here we divide the training examples (examples from chromosomes except val, test)
# into bound and unbound examples. In a later script we will sample balanced
# (half bound, half unbound) training datasets from these files.

# Construct the regex pattern
pattern=$(printf "(%s|%s)" "$(IFS='|'; echo "${stripped_val_chrs[*]}")" "$(IFS='|'; echo "${stripped_test_chrs[*]}")")

grep -Ev "$pattern[[:space:]]" "$allbed" | shuf > "$DATA_DIR/train_shuf.bed"
awk '$NF == 1' "$DATA_DIR/train_shuf.bed" | shuf > "$DATA_DIR/train_pos_shuf.bed"
awk '$NF == 0' "$DATA_DIR/train_shuf.bed" | shuf > "$DATA_DIR/train_neg_shuf.bed"

total_windows=`wc -l < "$DATA_DIR/train_shuf.bed"`
bound_windows=`wc -l < "$DATA_DIR/train_pos_shuf.bed"`
unbound_windows=`wc -l < "$DATA_DIR/train_neg_shuf.bed"`

total=$(( $bound_windows + $unbound_windows ))
if [[ $total != $total_windows ]]; then
	echo "Error: bound + unbound windows does not equal total windows. Exiting."
	exit 1
fi

echo "Bound training windows: $bound_windows"
echo "Unbound training windows: $unbound_windows"

echo "Done!"

exit 0