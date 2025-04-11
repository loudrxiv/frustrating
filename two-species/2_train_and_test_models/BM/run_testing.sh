#!/usr/bin/env bash

set -e

ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/two-species/tensorflow"

# Where all log files will be written to
# log files contain performance measurements
# made each epoch for the model.
log_root="$ROOT/logs"

# The python script below is also expecting a save directory for 
# models to exist
models_dir="$ROOT/models"

# The python script below is also expecting a save directory for 
# models to exist
output_dir="$ROOT/output"
mkdir -p "$output_dir"

# TFs to test models over
## "CTCF" "CEBPA" "HNF4A" "RXRA"
tfs=( "CTCF" "CEBPA" "HNF4A" "RXRA" ) 

# The target species to test the model on
## "mm10" "hg38"
sources=( "mm10" "hg38" )

# We extend to evaluate on all domains, not just mouse
## "mm10" "hg38"
domains=( "mm10" "hg38" )

if [ -f test_array.txt ]; then
	rm test_array.txt
	touch test_array.txt
fi

for tf in "${tfs[@]}"; do
    for source in "${sources[@]}"; do
        for domain in "${domains[@]}"; do
            printf "%s,%s,%s\n" $tf $source $domain >> test_array.txt
        done
    done
done

total_runs=$(cat test_array.txt | wc -l)

sbatch --array=1-${total_runs}%20 test.sh $ROOT $log_root