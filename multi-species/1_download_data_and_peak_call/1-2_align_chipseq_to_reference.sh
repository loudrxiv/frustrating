#!/usr/bin/env bash

#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=benos,dept_cpu,dept_gpu

set -e 

# Here we align the ChIP-seq data to the reference genome. We use bowtie2 to align the reads
# to the reference genome. We then filter out multi-mappers and duplicates.
#---------------------------------------------------------------------------------------------

ROOT=$1
genome=$2

RAW_DATA_DIR="$ROOT/raw_data"
CHIP_DATA_DIR="$ROOT/chip_seq_data"
BOWTIE_INDEX="$RAW_DATA_DIR/$genome/index/bowtie2"

# We choose to always activate `genomic_tools`
source activate genomic_tools

cd $CHIP_DATA_DIR

if [ $genome = "mm10" ]; then
    for i in $(ls do*mmu*.fq); do
        echo "Aligning $i to mm10..."

        # Run bowtie2
        bowtie2 -p 2 -q --local \
        -x "${BOWTIE_INDEX}/${genome}" \
        -U "${CHIP_DATA_DIR}/$i" \
        -S "${CHIP_DATA_DIR}/${i}.unsorted.sam"

        # Create BAM from SAM
        samtools view -h -S -b \
        -o "$i.unsorted.bam" \
        "$i.unsorted.sam"

        # Sort BAM file by genomic coordinates
        sambamba sort -t 2 \
        -o "$i.sorted.bam" \
        "$i.unsorted.bam"

        # Filter out multi-mappers and duplicates
        sambamba view -h -t 2 -f bam \
        -F "[XS] == null and not unmapped  and not duplicate" \
        "$i.sorted.bam" > "$i.final.bam"

        # Create indices for all the bam files for visualization and QC
        samtools index "$i.final.bam"
    done
elif [ $genome = "hg38" ]; then
    for i in $(ls do*hsa*.fq); do
        echo "Aligning $i to hg38..."

        # Run bowtie2
        bowtie2 -p 2 -q --local \
        -x "${BOWTIE_INDEX}/${genome}" \
        -U "${CHIP_DATA_DIR}/$i" \
        -S "${CHIP_DATA_DIR}/${i}.unsorted.sam"

        # Create BAM from SAM
        samtools view -h -S -b \
        -o "$i.unsorted.bam" \
        "$i.unsorted.sam"

        # Sort BAM file by genomic coordinates
        sambamba sort -t 2 \
        -o "$i.sorted.bam" \
        "$i.unsorted.bam"

        # Filter out multi-mappers and duplicates
        sambamba view -h -t 2 -f bam \
        -F "[XS] == null and not unmapped  and not duplicate" \
        "$i.sorted.bam" > "$i.final.bam"

        # Create indices for all the bam files for visualization and QC
        samtools index "$i.final.bam"
    done
elif [ $genome = "rheMac10" ]; then
    for i in $(ls do*mml*.fq); do
        echo "Aligning $i to rheMac10..."

        # Run bowtie2
        bowtie2 -p 2 -q --local \
        -x "${BOWTIE_INDEX}/${genome}" \
        -U "${CHIP_DATA_DIR}/$i" \
        -S "${CHIP_DATA_DIR}/${i}.unsorted.sam"

        # Create BAM from SAM
        samtools view -h -S -b \
        -o "$i.unsorted.bam" \
        "$i.unsorted.sam"

        # Sort BAM file by genomic coordinates
        sambamba sort -t 2 \
        -o "$i.sorted.bam" \
        "$i.unsorted.bam"

        # Filter out multi-mappers and duplicates
        sambamba view -h -t 2 -f bam \
        -F "[XS] == null and not unmapped  and not duplicate" \
        "$i.sorted.bam" > "$i.final.bam"

        # Create indices for all the bam files for visualization and QC
        samtools index "$i.final.bam"
    done
elif [ $genome = "canFam6" ]; then
    for i in $(ls do*cfa*.fq); do
        echo "Aligning $i to canFam6..."

        # Run bowtie2
        bowtie2 -p 2 -q --local \
        -x "${BOWTIE_INDEX}/${genome}" \
        -U "${CHIP_DATA_DIR}/$i" \
        -S "${CHIP_DATA_DIR}/${i}.unsorted.sam"

        # Create BAM from SAM
        samtools view -h -S -b \
        -o "$i.unsorted.bam" \
        "$i.unsorted.sam"

        # Sort BAM file by genomic coordinates
        sambamba sort -t 2 \
        -o "$i.sorted.bam" \
        "$i.unsorted.bam"

        # Filter out multi-mappers and duplicates
        sambamba view -h -t 2 -f bam \
        -F "[XS] == null and not unmapped  and not duplicate" \
        "$i.sorted.bam" > "$i.final.bam"

        # Create indices for all the bam files for visualization and QC
        samtools index "$i.final.bam"
    done
elif [ $genome = "rn7" ]; then
    for i in $(ls do*rno*.fq); do
        echo "Aligning $i to rn7..."

        # Run bowtie2
        bowtie2 -p 2 -q --local \
        -x "${BOWTIE_INDEX}/${genome}" \
        -U "${CHIP_DATA_DIR}/$i" \
        -S "${CHIP_DATA_DIR}/${i}.unsorted.sam"

        # Create BAM from SAM
        samtools view -h -S -b \
        -o "$i.unsorted.bam" \
        "$i.unsorted.sam"

        # Sort BAM file by genomic coordinates
        sambamba sort -t 2 \
        -o "$i.sorted.bam" \
        "$i.unsorted.bam"

        # Filter out multi-mappers and duplicates
        sambamba view -h -t 2 -f bam \
        -F "[XS] == null and not unmapped  and not duplicate" \
        "$i.sorted.bam" > "$i.final.bam"

        # Create indices for all the bam files for visualization and QC
        samtools index "$i.final.bam"
    done
fi