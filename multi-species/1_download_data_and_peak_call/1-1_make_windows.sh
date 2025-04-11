#!/usr/bin/env bash

#SBATCH --partition=benos,dept_cpu,dept_gpu
#SBATCH --job-name=%x_%j
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

# Here we tile the genome into windows of a fixed size and stride. We then filter out any
# windows that contain Ns and blacklist regions.
#---------------------------------------------------------------------------------------------

RAW_DATA_DIR=$1
GENOME=$2
GENOME_FILE=$3
WINDOW_SIZE=$4
WINDOW_STRIDE=$5

# We choose to always activate `genomic_tools`
source activate genomic_tools

# We don't have blacklists for every species, but we do have them
# for human and mouse genomes. We can pretend for now that every
# species has them, but this will be ignored for rheMac10, canFam6
# and rn7.
GENOME_DATA_DIR="${RAW_DATA_DIR}/${GENOME}"
BLACKLIST_BED_FILE="${RAW_DATA_DIR}/${GENOME}/${GENOME}.blacklist.bed"

#--- 1. Gets the sequence for each window in the bed file, returned in bed format
# This script assumes the chromosome sizes file chrom.sizes is in $RAW_DATA_DIR
# The output is a bed file called windows.unfiltered.bed

echo "Creating windows files for ${GENOME} genome."

python 1-1-1_make_windows_bed.py "$RAW_DATA_DIR" "$GENOME" "$WINDOW_SIZE" "$WINDOW_STRIDE"

#--- 2. Filters out any lines with N (any bases in the genome that are unknown)

if [ -f "${GENOME_DATA_DIR}/windows.noN.bed" ]; then
    printf "We have the file that represents the windows with no Ns! \n"
else 
    echo "Getting genomic sequences for all regions to filter unresolved sequence regions..."
    bedtools getfasta -fi "$GENOME_FILE" -bed "${GENOME_DATA_DIR}/windows.unfiltered.bed" -bedOut | grep -v "n" | grep -v "N" | awk -v OFS="\t" '{print $1, $2, $3 }' | sort -k1,1 -k2,2n > "${GENOME_DATA_DIR}/windows.noN.bed" || exit 1
fi

#--- 3. Removes the blacklisted regions from the bed file of the hg38 and mm10 genomes

if [ -f "${GENOME_DATA_DIR}/windows.bed" ]; then
    printf "We have the bed file that represents the windows! \n"
else 
    if [ $GENOME = "hg38" ] || [ $GENOME = "mm10" ]; then
        echo "Filtering out ENCODE blacklist regions from bed file..."
        bedtools intersect -a "${GENOME_DATA_DIR}/windows.noN.bed" -b "$BLACKLIST_BED_FILE" -v > "${GENOME_DATA_DIR}/windows.bed" || exit 1
    else
        cp "${GENOME_DATA_DIR}/windows.noN.bed" "${GENOME_DATA_DIR}/windows.bed"
    fi
fi

printf "Done.\n\n"

exit 0