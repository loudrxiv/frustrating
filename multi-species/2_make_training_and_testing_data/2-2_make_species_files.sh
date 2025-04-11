#!/usr/bin/env bash

#---------------------------------------------------------------------------------------------------
# We create a shuffled species-background training dataset for a given TF and species.
#---------------------------------------------------------------------------------------------------

set -e

ROOT=$1
tf=$2
genome=$3

echo "Prepping shuffled species-background datasets for $tf in genome $genome."

DATA_DIR="$ROOT/data/$genome/$tf"
TRAIN_FILE="$DATA_DIR/train_shuf.bed"
POS_FILE="$DATA_DIR/train_pos_shuf.bed"

# Get the number of bound windows
bound_windows=`wc -l < "$POS_FILE"`

# Process of getting distinct randomly selected examples.

tmp_shuf_file="$DATA_DIR/chr3toY_shuf.tmp"

shuf "$TRAIN_FILE" > "$tmp_shuf_file"

background_filename="$DATA_DIR/train_background_shuf.bed"
cat "$tmp_shuf_file" > "$background_filename"

rm "$tmp_shuf_file"

echo "Done!"

exit 0