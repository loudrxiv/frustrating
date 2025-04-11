#!/usr/bin/env bash

#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=benos,dept_cpu,dept_gpu
#SBATCH --exclusive

set -e 

# This script uses the output from make_windows_file.sh as well as peak calls
# to generate the full dataset for a given species and TF.
# The output is a file called all.all in the raw_data directory.
# The all.all file contains all of the windows that have survived filtering,
# to potentially be part of the train/val/test data for the model.
# It is bed-formatted, with one additional column containing the binary label
# where 1 corresponds to "a peak overlapped with this window" and 0
# corresponds to "a peak did not overlap with this window".

# IMPORTANT -- once you run this script, if everything went okay, you need to
# move the all.all file from raw_data/${species}/${tf} to data/${species}/${tf}.
# I make you do this to avoid you accidentally overwriting your master dataset file.
#--------------------------------------------------

ROOT=$1
genome=$2
tf=$3

DATA_DIR="$ROOT/raw_data/${genome}/${tf}"
BLACKLIST_FILE="$ROOT/raw_data/${genome}/${genome}.blacklist.bed"

# The script 0.0_make_windows_files.sh created these files
WINDOWS_FILE="$ROOT/raw_data/${genome}/windows.bed"

echo "Using genome ${genome} and TF ${tf}."

# We choose to always activate `genomic_tools`
source activate genomic_tools

#--- (1)
# This is where we ge the lables for the TF binding, you NEED
# the output from whatever peak calling method you used to get this.

echo "Making TF labels..."
./1-3-1_make_tf_labels.sh "$ROOT" "$genome" "$tf"

#--- (2)
# combine the bed-formatted columns for windows and the single column of
# binding labels into one file

echo "Generating full dataset..."

# Output of the script above
tf_labels_file="$DATA_DIR/binding_labels.bed"

paste -d "	" "$WINDOWS_FILE" "$tf_labels_file" > "$DATA_DIR/all.bedsort.tmp.all"
if [ ! -s "$DATA_DIR/all.bedsort.tmp.all" ]; then
  echo "Error: failed at paste command."
  exit 1
fi

# Remove any windows that intersect with blacklist regions or unmappable regions. Of note, we
# only do this for genomes that have blacklist files available
if [ $genome = "hg38" ] || [ $genome = "mm10" ]; then
  bedtools intersect -v -a "$DATA_DIR/all.bedsort.tmp.all" -b "$BLACKLIST_FILE" > "$DATA_DIR/all.noBL.tmp.all"
  if [ ! -s "$DATA_DIR/all.noBL.tmp.all" ]; then
    echo "Error: failed at blacklist intersect command."
    exit 1
  fi
else
  cp "$DATA_DIR/all.bedsort.tmp.all" "$DATA_DIR/all.noBL.tmp.all"
fi

# Finally, remove weird chromosomes and fix the file formatting
# no chrM, no chrEBV, and no scaffolds will be used
# specifically, this line removes a redundant set of bed-info columns (chr \t start \t stop) in a slightly hacky way
grep -E "chr[0-9]+" "$DATA_DIR/all.noBL.tmp.all" | sed -E 's/	/:/' | sed -E 's/	/-/' | sed -E 's/chr[0-9]+	[0-9]+	[0-9]+	//g' | sed -E 's/[:-]/	/g'> "$DATA_DIR/all.all"
if [ ! -s "$DATA_DIR/all.all" ]; then
  echo "Error: failed at final step."
  exit 1
fi

# Cleanup -- delete tmp files
rm "$DATA_DIR/all.bedsort.tmp.all" "$DATA_DIR/all.noBL.tmp.all" "$tf_labels_file"

lines=$(wc -l < "$DATA_DIR/all.all")

echo "Done! Whole genome file (all.all) contains ${lines} windows."

exit 0