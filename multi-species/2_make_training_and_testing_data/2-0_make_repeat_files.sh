#!/usr/bin/env bash

set -e 

# NOTE: This script needs to know which windows are in the dataset -- 
# it can figure that out from any "all.all" file that is the result of 
# all the data pre-processing that comes before this point.
#
# So, you need to pass in the path to /any/ all.all file (it doesn't matter
# which tf)
#
# Some analyses will require either a file containing all annotated repeat
# elements in the test chromosome. This script will create those files.
#---------------------------------------------------------------------------------------------------

root=$1
genome=$2
allall_file=$3

# Reconstruct the data root
DATA_ROOT="$root/data/$genome"

#--- (2) Isolate SINEs and subtypes from the full RepeatMasker file
# Each genome has a RepeatMasker file that lists all repeat elements in the genome, but the 
# names of them differ between genomes. We need to isolate the SINEs and their subtypes
# from the RepeatMasker file for the genome we're working with.

# SINEs
awk -v OFS="\t" '{ if ($12 == "SINE") print $6, $7, $8, $11, $12, $13 }' "$DATA_ROOT/rmsk.bed" > "$DATA_ROOT/sines.bed"

# SINE subfamilies
if [ $genome = "mm10" ]; then
	# B1s are a subfamily of SINEs in the mouse genome that are labelled as "Alu"
	awk -v OFS="\t" '{ if ($13 == "Alu") print $6, $7, $8, $11, $12, $13 }' "$DATA_ROOT/rmsk.bed" > "$DATA_ROOT/b1s.bed"
elif [ $genome = "hg38" ]; then
	awk -v OFS="\t" '{ if ($13 == "Alu") print $6, $7, $8, $11, $12, $13 }' "$DATA_ROOT/rmsk.bed" > "$DATA_ROOT/alus.bed"
elif [ $genome = "rheMac10" ]; then
	awk -v OFS="\t" '{ if ($13 == "Alu") print $6, $7, $8, $11, $12, $13 }' "$DATA_ROOT/rmsk.bed" > "$DATA_ROOT/alus.bed"
elif [ $genome = "canFam6" ]; then
	printf "We will use SINE annotations here..."
elif [ $genome = "rn7" ]; then
	awk -v OFS="\t" '{ if ($13 == "Alu") print $6, $7, $8, $11, $12, $13 }' "$DATA_ROOT/rmsk.bed" > "$DATA_ROOT/b1s.bed"
fi

echo "Done."

exit 0