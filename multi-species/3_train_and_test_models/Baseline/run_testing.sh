#!/usr/bin/env bash

set -e

ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/multi-species"

# TFs to train models over. We have the following options:
## "CEBPA" "FOXA1" "HNF4A" "HNF6"
tfs=( "CEBPA" "FOXA1" "HNF4A" "HNF6" ) 

# The target species to train the model on. We have the
# following options:
## "mm10" "rn7" "rheMac10" "canFam6" "hg38"
targets=( "mm10" "rn7" "rheMac10" "canFam6" "hg38" )

if [ -f test_array.txt ]; then
	rm test_array.txt
	touch test_array.txt
fi

# We will test a model for each TF on each target species
for tf in "${tfs[@]}"; do
	for target in "${targets[@]}"; do
		printf "%s,%s\n" $tf $target >> test_array.txt
    done
done

total_runs=$(cat test_array.txt | wc -l)

sbatch --array=1-${total_runs}%20 test.sh $ROOT