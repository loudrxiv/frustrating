#!/usr/bin/env bash

# Reorganizing processed data!

set -e
#-----------------------------------------------------------

TFS=( "CTCF" "CEBPA" "HNF4A" "RXRA" )
GENOMES=( "mm10" "hg38" )

# We check if we need to unzip processed data
if [ -d kelly_paper_peaks ]; then
    printf "Peaks already unzipped\n"
else
    # (1) Unzip the peaks
    unzip kelly_paper_peaks.zip

    # (2) Start moving information into the correct directories
    for i in "${TFS[@]}"; do
        for j in "${GENOMES[@]}"; do
            mv "$(pwd)/kelly_paper_peaks/${i}/${i}_liver_${j}.bed" "raw_data/${j}/${i}/mgps_out_${i}.bed"
        done
    done

fi
