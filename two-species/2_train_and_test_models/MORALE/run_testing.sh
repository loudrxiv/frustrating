#!/usr/bin/env bash

set -e

ROOT="/net/talisker/home/benos/mae117/Documents/research/dennis/domain_adaptation/RECOMB/two-species/tensorflow"

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

			if [ "$source" == "hg38" ]; then
				if [ "$tf" == "CEBPA" ]; then
					lambda_="7.0"
					match="1"
					# 0.2797990475578469, 0.2875005810345951
				elif [ "$tf" == "CTCF" ]; then
					lambda_="4.0"
					match="1"
					# 0.6102941138477869, 0.6990186674550396
				elif [ "$tf" == "HNF4A" ]; then
					lambda_="8.0"
					match="1"
					# 0.24424882915776389, 0.2764845806329447
				elif [ "$tf" == "RXRA" ]; then
					lambda_="8.0"
					match="0"
					# 0.21038397084229232, 0.3263132161625715
				else
					printf "Invalid TF: $tf\n"
					exit 1
				fi
			elif [ "$source" == "mm10" ]; then
				if [ "$tf" == "CEBPA" ]; then
					lambda_="8.0"
					match="0"
					# 0.24863538855972514, 0.36616049155507213
				elif [ "$tf" == "CTCF" ]; then
					lambda_="4.0"
					match="0"
					# 0.6458876708489377, 0.7069821656079107
				elif [ "$tf" == "HNF4A" ]; then
					lambda_="6.0"
					match="0"
					# 0.21674723512185973, 0.28873938750755773
				elif [ "$tf" == "RXRA" ]; then
					lambda_="7.0"
					match="0"
					# 0.20542190826940987, 0.24081767937478246
				else
					printf "Invalid TF: $tf\n"
					exit 1
				fi
			else 
				printf "Invalid source: $source\n"
				exit 1
			fi

			# Now write line to test_array.txt
			printf "%s,%s,%s,%s,%s\n" $tf $source $match $lambda_ $domain >> test_array.txt

        done
    done
done

total_runs=$(cat test_array.txt | wc -l)

sbatch --array=1-${total_runs}%20 test.sh $ROOT