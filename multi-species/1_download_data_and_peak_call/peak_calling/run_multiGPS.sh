#!/usr/bin/env bash

#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=benos,dept_cpu,dept_gpu
#SBATCH --exclusive

ROOT=$1
genome=$2
tf=$3

# https://mahonylab.org/software/multigps/
MULTIGPS_PATH="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/githubs/multigps"

# multiGPS options
DESIGN_FILE="${ROOT}/1_download_data_and_peak_call/peak_calling/${genome}.${tf}.design"

CHIPSEQ_DATA_ROOT="${ROOT}/chip_seq_data"
GENOME_DIR="${ROOT}/raw_data/${genome}"
SIZES_FILE="${GENOME_DIR}/${genome}.fa.sizes"
REF_FILE="${GENOME_DIR}/${genome}.fa"
OUT_DIR="${GENOME_DIR}/${tf}/mgps_out"

# This file is a bed file of regions to not call peaks in
# Blacklist is taken from ENCODE (Amemiya et al. 2019) for hg38 and mm10.
# This doesn't exist for other genomes, so we have two different peak call, 
# well, calls.
BLACKLIST_PATH="${GENOME_DIR}/${genome}.blacklist.bed"

printf "Running multiGPS...\n\n"

if [ $genome = "hg38" ]; then
    printf "=== hg38 === \n"
    java -Xmx45G -jar "${MULTIGPS_PATH}/multigps.v0.75.jar" --geninfo "${SIZES_FILE}" --seq "${REF_FILE}" --design "${DESIGN_FILE}" --out "${OUT_DIR}" --verbose --threads 8 --exclude "${BLACKLIST_PATH}"
elif [ $genome = "mm10" ]; then
    printf "=== mm10 === \n"
    java -Xmx45G -jar "${MULTIGPS_PATH}/multigps.v0.75.jar" --geninfo "${SIZES_FILE}" --seq "${REF_FILE}" --design "${DESIGN_FILE}" --out "${OUT_DIR}" --verbose --threads 8 --exclude "${BLACKLIST_PATH}"
elif [ $genome = "rheMac10" ]; then
    printf "=== rheMac10 === \n"
    java -Xmx45G -jar "${MULTIGPS_PATH}/multigps.v0.75.jar" --geninfo "${SIZES_FILE}" --seq "${REF_FILE}" --design "${DESIGN_FILE}" --out "${OUT_DIR}" --verbose --threads 8
elif [ $genome = "canFam6" ]; then
    printf "=== canFam6 === \n"
    java -Xmx45G -jar "${MULTIGPS_PATH}/multigps.v0.75.jar" --geninfo "${SIZES_FILE}" --seq "${REF_FILE}" --design "${DESIGN_FILE}" --out "${OUT_DIR}" --verbose --threads 8
elif [ $genome = "rn7" ]; then
    printf "=== rn7 === \n"
    java -Xmx45G -jar "${MULTIGPS_PATH}/multigps.v0.75.jar" --geninfo "${SIZES_FILE}" --seq "${REF_FILE}" --design "${DESIGN_FILE}" --out "${OUT_DIR}" --verbose --threads 8
else
    echo "Genome not supported"
    exit 1
fi

cp "${OUT_DIR}/mgps_out_${tf}.bed" "${OUT_DIR}/../"

printf "Done\n"