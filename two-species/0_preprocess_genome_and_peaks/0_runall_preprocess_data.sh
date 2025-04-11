#!/usr/bin/env bash

#-----------------------------------------------------------------------------
# This script pre-processes information from the genomes of both species
# and from the peak calls for each TF within each species to create a file
# of the genome-wide dataset for each TF-species combo. This file will then
# be used by scripts in the 1_* directory for model-specific training/val/test
# data preprocessing. See inside the scripts called here for more explanation.
#-----------------------------------------------------------------------------

set -e

ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/two-species/tensorflow"

TFS=( "CTCF" "CEBPA" "HNF4A" "RXRA" )
GENOMES=( "mm10" "hg38" )

for genome in "${GENOMES[@]}"; do

  # You'll need to replace these paths with paths to your own genome fastas!
  if [ "$genome" = "mm10" ] ; then
    GENOME_FILE="${ROOT}/raw_data/mm10/mm10_no_alt_analysis_set_ENCODE.fasta"

    # Grab it if we don't have it! and put it in a place that makes sense
    if [ -f "${ROOT}/raw_data/mm10/${GENOME_FILE}" ]; then 
      echo "We have the genome file we need! No neeed to fetch it..."
    else
      wget https://www.encodeproject.org/files/mm10_no_alt_analysis_set_ENCODE/@@download/mm10_no_alt_analysis_set_ENCODE.fasta.gz
      gunzip mm10_no_alt_analysis_set_ENCODE.fasta.gz
      mv mm10_no_alt_analysis_set_ENCODE.fasta "${ROOT}/raw_data/mm10/"
    fi
  else # hg38
    GENOME_FILE="${ROOT}/raw_data/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"

    # Grab it if we don't have it! and put it in a place that makes sense
    if [ -f "${ROOT}/raw_data/hg38/${GENOME_FILE}" ]; then 
      echo "We have the genome file we need! No neeed to fetch it..."
    else
      wget https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz
      gunzip GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz
      mv GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta "${ROOT}/raw_data/hg38/"
    fi
  fi

  ./0.0_make_windows_files.sh "$ROOT" "$genome" "$GENOME_FILE"

  for tf in "${TFS[@]}"; do
    # You need the peak calls for each TF in each species here before this step!
    ./0.1_create_full_dataset.sh "$ROOT" "$genome" "$tf"    
  done
  
done