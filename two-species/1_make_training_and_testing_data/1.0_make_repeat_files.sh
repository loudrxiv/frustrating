#!/usr/bin/env bash

#-------------------------------------------------------------------------------
# NOTE: This script needs to know which windows are in the dataset -- 
# it can figure that out from any "all.all" file that is the result of 
# all the data pre-processing that comes before this point.
# So, you need to pass in the path to /any/ all.all file (it doesn't matter
# which tf) for hg38/mm10, and the script will deal with the rest.
# The all.all files are created by the script
# ../0_preprocess_genome_and_peaks/0.1_create_full_dataset.sh,
# (if you ran ../0_preprocess_genome_and_peaks/0_runall.sh, that script
# would have run the 0.1 script, so that works too).

# NOTE: This script assumes chromosome 2 for hg38/mm10 is the test set.
# Modify the commands below if that is not the case.

# Some analyses will require either a file containing all annotated Alu
# repeats (chr2_alus.bed) or a file containing all genomic windows that
# overlap with any Alu repeat (chr2_alus_intersect.bed). This script will
# create those files.

# This script also creates a sines.bed file -- a similar line of code was
# used to create corresponding files for other repeat types (e.g. LINEs)
# for the analyses done for Supplementary Table 1 (see the relevant jupyter
# notebook for generating that table).
#-------------------------------------------------------------------------------

set -e 

ROOT=$1
allall_file=$2
genome=$3
DATA_ROOT="${ROOT}/data/${genome}"

# Step 1: download rmsk.bed from UCSC Table Browser of tracks (RepeatMasker track)
# this is done for you in the script ../setup_directories_and_download_files.sh
# Confirm with your eyeballs that the columns look correct (see awk below)
if [[ ! -f "$DATA_ROOT/rmsk.bed" ]]; then
	echo "RepeatMasker track needs to be downloaded from UCSC."
	exit 1
fi

if [[ "$genome" == "hg38" ]]; then
	echo "Prepping repeat files for $genome."

	# Step 2: isolate SINEs and Alus from the full RepeatMasker file
	awk -v OFS="\t" '{ if ($12 == "SINE") print $6, $7, $8, $11, $12, $13 }' "$DATA_ROOT/rmsk.bed" > "$DATA_ROOT/sines.bed"
	awk -v OFS="\t" '{ if ($13 == "Alu") print $6, $7, $8, $11, $12, $13 }' "$DATA_ROOT/rmsk.bed" > "$DATA_ROOT/alus.bed"

	# Then isolate just the Alus that are on chromosome 2, the test chromosome
	# (having this file speeds up later analyses that are restricted to the test data)
	awk -v OFS="\t" '$1 == "chr2"' "$DATA_ROOT/alus.bed" > "$DATA_ROOT/chr2_alus.bed"

	### Step 3: filter chr2 Alus for only the ones that overlap at all with windows in our dataset
	# these are the only Alus we'll care about in downstream analysis
	if [[ ! -f "$DATA_ROOT/chr2.bed" ]]; then
		awk -v OFS="\t" '{ if ($1 == "chr2") print $1, $2, $3 }' "$allall_file" > "$DATA_ROOT/chr2.bed"  
	fi

	bedtools intersect -a "$DATA_ROOT/chr2.bed" -b "$DATA_ROOT/chr2_alus.bed" -u > "$DATA_ROOT/chr2_alus_intersect.bed"
else
	echo "Prepping repeat files for $genome."

	### Step 2: isolate SINEs, Alu, and B1 element from the full RepeatMasker file
	awk -v OFS="\t" '{ if ($12 == "SINE") print $6, $7, $8, $11, $12, $13 }' "$DATA_ROOT/rmsk.bed" > "$DATA_ROOT/sines.bed"
	awk -v OFS="\t" '{ if ($13 == "Alu") print $6, $7, $8, $11, $12, $13 }' "$DATA_ROOT/rmsk.bed" > "$DATA_ROOT/b1s.bed"

	# Then isolate those elements that are only on chromosome 2, the test chromosome
	# (having this file speeds up later analyses that are restricted to the test data)
	awk -v OFS="\t" '$1 == "chr2"' "$DATA_ROOT/b1s.bed" > "$DATA_ROOT/chr2_b1s.bed"

	### Step 3: filter chr2 B2s for only the ones that overlap at all with windows in our dataset
	# these are the only Alus we'll care about in downstream analysis
	if [[ ! -f "$DATA_ROOT/chr2.bed" ]]; then
		awk -v OFS="\t" '{ if ($1 == "chr2") print $1, $2, $3 }' "$allall_file" > "$DATA_ROOT/chr2.bed"  
	fi

	bedtools intersect -a "$DATA_ROOT/chr2.bed" -b "$DATA_ROOT/chr2_b1s.bed" -u > "$DATA_ROOT/chr2_b1s_intersect.bed"
fi

echo "Done."

exit 0