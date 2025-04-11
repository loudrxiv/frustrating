#!/usr/bin/env bash

#-----------------------------------------------------------------------------
# This script creates the directories that will be used by subsequent scripts.
# It also downloads a few files needed in the other scripts.
#-----------------------------------------------------------------------------

set -e

# This is the directory you want all of the project to be contained in
ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/two-species/tensorflow"
RAW_DATA_DIR="$ROOT/raw_data"

# We ensure the raw data directory exists
mkdir -p "$RAW_DATA_DIR"

# Directories will be made for each species, and for each TF within each species directory
TFS=( "CTCF" "CEBPA" "HNF4A" "RXRA" )
GENOMES=( "mm10" "hg38" )

for genome in "${GENOMES[@]}"; do
  genome_dir="$RAW_DATA_DIR/$genome"
  mkdir -p "$genome_dir"

  for tf in "${TFS[@]}"; do
    tf_dir="$genome_dir/$tf"
    mkdir -p "$tf_dir"
  done

done

# Where data will be once it is processed, as it is being prepped for model training/testing
PROCESSED_DATA_DIR="$ROOT/data"

# Directory structure is identical to that of the raw data 
cp -r "$RAW_DATA_DIR" "$PROCESSED_DATA_DIR"

# download and unzip ENCODE blacklist files, UMAP tracks, and chromosome size files

# for hg38:

cd "$RAW_DATA_DIR/hg38"
wget http://mitra.stanford.edu/kundaje/akundaje/release/blacklists/hg38-human/hg38.blacklist.bed.gz
gunzip hg38.blacklist.bed.gz

wget https://bismap.hoffmanlab.org/raw/hg38/k36.umap.bed.gz
gunzip k36.umap.bed.gz

wget "http://genome.ucsc.edu/goldenPath/help/hg38.chrom.sizes" -O "chrom.sizes"
grep -v "_" "chrom.sizes" | grep -v "X" | grep -v "Y" | grep -v "M" > tmp
mv tmp "chrom.sizes"

cd "$PROCESSED_DATA_DIR/hg38"
wget https://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz
gunzip rmsk.txt.gz
mv rmsk.txt rmsk.bed


# for mm10:

cd "$RAW_DATA_DIR/mm10"
wget http://mitra.stanford.edu/kundaje/akundaje/release/blacklists/mm10-mouse/mm10.blacklist.bed.gz
gunzip mm10.blacklist.bed.gz

wget https://bismap.hoffmanlab.org/raw/mm10/k36.umap.bed.gz
gunzip k36.umap.bed.gz

wget "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes" -O "chrom.sizes"
grep -v "_" "chrom.sizes" | grep -v "X" | grep -v "Y" | grep -v "M" > tmp
mv tmp "chrom.sizes"


cd "$PROCESSED_DATA_DIR/mm10"
wget https://hgdownload.cse.ucsc.edu/goldenPath/mm10/database/rmsk.txt.gz
gunzip rmsk.txt.gz
mv rmsk.txt rmsk.bed

echo "Done."

exit 0