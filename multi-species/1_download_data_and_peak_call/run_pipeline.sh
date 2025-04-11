#!/usr/bin/env bash

set -e

# This is the master script that will run the preprocessing pipeline from start to finish for 
# this analysis. 
#---------------------------------------------------------------------------------------------

# Grab the step the user calls in the pipeline to process the data
STEP=$1

# We don't accept this from the user. This is static. 
ROOT="$(pwd)../"
RAW_DATA_DIR="$ROOT/raw_data"

# Other statics we need for the variety of operations
TFS=( "HNF6" "CEBPA" "FOXA1" "HNF4A" )
GENOMES=( "hg38" "mm10" "rheMac10" "canFam6" "rn7" )
WINDOW_SIZE=1000
WINDOW_OVERLAP=50

# We choose to always activate `genomic_tools`
source activate genomic_tools

#--- (0+1) Our first steps in the pipeline! Running `1_generate_references` does the following:
# First, we grab the reference genomes of each species under examination, if not already present
# (this script assumes you have setup the directories). Afterwards, we divide the references into
# a set of windows with a predetermined window size in the set of static variables above. These windows 
# then need to be aligned to the reference of each genome.
if [ $STEP = "1_generate_data" ]; then
    for genome in "${GENOMES[@]}"; do

      # Generate references, indices, and blacklists for specified genome
      gen_jobid=$(sbatch --job-name=gen_data 1-0_generate_all_data.sh "$RAW_DATA_DIR" "$genome" | awk '{print $4}')

      # Point to the reference genome
      if [ $genome = "mm10" ]; then
        GENOME_FILE="${RAW_DATA_DIR}/mm10/mm10.fa"
      elif [ $genome = "hg38" ]; then
        GENOME_FILE="${RAW_DATA_DIR}/hg38/hg38.fa"
      elif [ $genome = "rheMac10" ]; then
        GENOME_FILE="${RAW_DATA_DIR}/rheMac10/rheMac10.fa"
      elif [ $genome = "canFam6" ]; then
        GENOME_FILE="${RAW_DATA_DIR}/canFam6/canFam6.fa"
      elif [ $genome = "rn7" ]; then
        GENOME_FILE="${RAW_DATA_DIR}/rn7/rn7.fa"
      fi

      # Create the windows for each species (we run these after all the data is in place)
      sbatch --job-name=make_windows --dependency=afterok:"$gen_jobid" 1-1_make_windows.sh "$RAW_DATA_DIR" "$genome" "$GENOME_FILE" "$WINDOW_SIZE" "$WINDOW_OVERLAP"

    done
#--- (2)
# Now that we have generated the references and indicies for each genome, we can align the
# chipseq data to the reference genome. This is done in parallel for each genome.
elif [ $STEP = "2_align_to_reference" ]; then
  for genome in "${GENOMES[@]}"; do
    printf "=== ${genome} === \n"
    sbatch --job-name=align_chipseq_to_reference 1-2_align_chipseq_to_reference.sh "$ROOT" "$genome"
  done
#--- (3)
# We have aligned the chipseq data to the reference genome. Now we can call peaks for each TF
# in parallel for each genome.
elif [ $STEP = "3_call_peaks" ]; then
  for genome in "${GENOMES[@]}"; do
    for tf in "${TFS[@]}"; do
      sbatch --job-name="peakcaller_${genome}.${tf}" peak_calling/run_multiGPS.sh $ROOT $genome $tf
    done
  done
#--- (4)
# Finally, we can create the all.all files for each TF in parallel for each genome.
elif [ $STEP = "4_create_all-alls" ]; then # NOTE: Can only run one tf at a time?? Yeee...i think this needs to run (tf-wise) sequentially
  for genome in "${GENOMES[@]}"; do
    for tf in "${TFS[@]}"; do
      
      # Inside each nested directory, all transcription factors should have
      # the output from peak calling, otherwise you'll get errors related to
      # missing files 
      printf "=== creating all.all for ${genome}.${tf} === \n"

      ./1-3_create_full_dataset.sh $ROOT $genome $tf > "create_all-alls.${genome}.${tf}.out" &
      
      process_id=$!
      echo "PID: $process_id"
      wait $process_id
      echo "Exit status: $?"
      
    done
  done
else
  printf "Invalid step. This should lowkey not get raised unless you directly call the bash script...\n"
fi