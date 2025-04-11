#!/usr/bin/env bash

set -e 

# This script needs to be run first in order to create the directories that will be used by
# subsequent scripts. It initally downloads a the data we are to be working with and a few
# other files needed for the entire process.
#--------------------------------------------------------------------------------------------

ROOT=$(pwd)
RAW_DATA_DIR="$ROOT/raw_data"

mkdir -p "$RAW_DATA_DIR"

# Directories will be made for each species, and for each TF within each species directory
TFS=( "CEBPA" "HNF4A" "HNF6" "FOXA1" )
GENOMES=( "mm10" "hg38" "rheMac10" "canFam6" "rn7" )

for genome in "${GENOMES[@]}"; do

  genome_dir="$RAW_DATA_DIR/$genome"
  mkdir -p "$genome_dir"

  for tf in "${TFS[@]}"; do
    tf_dir="$genome_dir/$tf"
    mkdir -p "$tf_dir"
    mkdir -p "$tf_dir/mgps_out"
  done

done

# Where data will be once it is processed, as it is being prepped for model training/testing
PROCESSED_DATA_DIR="$ROOT/data"

# Directory structure is identical to that of the raw data 
cp -r "$RAW_DATA_DIR" "$PROCESSED_DATA_DIR"

# Now we download all of the respective files for all of the species we are working with...

#--- hg38

printf "\n==== hg38 ====\n\n"
cd "$RAW_DATA_DIR/hg38"

if [ ! -f "${PROCESSED_DATA_DIR}/hg38/rmsk.bed" ]; then
  printf "Downloading hg38 rmsk.bed\n"
  cd "$PROCESSED_DATA_DIR/hg38"
  wget https://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz
  gunzip rmsk.txt.gz
  mv rmsk.txt rmsk.bed
else
  printf "hg38 rmsk.bed already exists"
fi

#--- mm10

printf "\n\n==== mm10 ====\n\n"
cd "$RAW_DATA_DIR/mm10"

if [ ! -f "${PROCESSED_DATA_DIR}/mm10/rmsk.bed" ]; then
  printf "Downloading mm10 rmsk.bed\n"
  cd "$PROCESSED_DATA_DIR/mm10"
  wget https://hgdownload.cse.ucsc.edu/goldenPath/mm10/database/rmsk.txt.gz
  gunzip rmsk.txt.gz
  mv rmsk.txt rmsk.bed
else
  printf "mm10 rmsk.bed already exists"
fi

#--- rheMac10
# For the Macaca mulatta (Mmul_10), or Rhesus monkey (rheMac10), there does not exist a blacklist file
# we can just pull and use. The blacklist files are from a work that only looked at mouse, human,
# worm, and fly genomes [1, 2].

# I think this is okay not to use, i.e., to remove those presumed regions. They target unannotated 
# repeat elements usually (?). We move forward without getting it.

# [1]: https://genome.cshlp.org/content/32/3/512.full#sec-1
# [2]: https://www.nature.com/articles/s41598-019-45839-z
#
# This is the same story for the umap/bismap mappability, which is only available in human
# and mouse genomes [3].
#
# [3]: https://academic.oup.com/nar/article/46/20/e120/5086676

printf "\n\n==== rheMac10 ====\n\n"
cd "$RAW_DATA_DIR/rheMac10"

if [ ! -f "${PROCESSED_DATA_DIR}/rheMac10/rmsk.bed" ]; then
  printf "Downloading rheMac10 rmsk.bed\n"
  cd "$PROCESSED_DATA_DIR/rheMac10"
  wget https://hgdownload.cse.ucsc.edu/goldenPath/rheMac10/database/rmsk.txt.gz
  gunzip rmsk.txt.gz
  mv rmsk.txt rmsk.bed
else
  printf "rheMac10 rmsk.bed already exists"
fi

#--- canFan6
# See above for why we don't include a blacklist/bismap

printf "\n\n==== canFam6 ====\n\n"
cd "$RAW_DATA_DIR/canFam6"

if [ ! -f "${PROCESSED_DATA_DIR}/canFam6/rmsk.bed" ]; then
  printf "Downloading canFam6 rmsk.bed\n"
  cd "$PROCESSED_DATA_DIR/canFam6"
  wget https://hgdownload.cse.ucsc.edu/goldenPath/canFam6/database/rmsk.txt.gz
  gunzip rmsk.txt.gz
  mv rmsk.txt rmsk.bed
else
  printf "canFam6 rmsk.bed already exists"
fi

#--- rn7
# See above for why we don't include a blacklist/bismap

printf "\n\n==== rn7 ====\n\n"
cd "$RAW_DATA_DIR/rn7"

if [ ! -f "${PROCESSED_DATA_DIR}/rn7/rmsk.bed" ]; then
  printf "Downloading rn7 rmsk.bed\n"
cd "$PROCESSED_DATA_DIR/rn7"
wget https://hgdownload.cse.ucsc.edu/goldenPath/rn7/database/rmsk.txt.gz
gunzip rmsk.txt.gz
mv rmsk.txt rmsk.bed
else
  printf "rn7 rmsk.bed already exists\n\n"
fi

# Now get the chip-seq data!
${ROOT}/ena-file-download-read_run-PRJEB1571-submitted_ftp-20240816-1712.sh ${ROOT}

echo "Done."
exit 0