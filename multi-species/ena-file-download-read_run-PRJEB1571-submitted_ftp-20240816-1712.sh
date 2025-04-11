#!/usr/bin/env bash

set -e

# All used ChIP-seq files used  were deposited here under the ArrayExpress
# accession number of E-MTAB-1509 [1,2,3]. Keep in mind that the authors filled
# genomic gaps in the coding regions for dog FOXA1 and macaque ONECUT1 and 
# deposited them under NCBI accessions numbers JN601139 and JQ178331 respectively.

# [1]: https://elifesciences.org/articles/02626#data
# [2]: https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-1509?query=E-MTAB-1509
# [3]: https://www.ebi.ac.uk/ena/browser/view/PRJEB1571
#-----------------------------------------------------------------------------------------

ROOT=$1

# Create and move into ChIP-seq directory
mkdir -p "${ROOT}/chip_seq_data/"
cd "${ROOT}/chip_seq_data"

# Download all files we need
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235788/do401_Input_liver_none_hsaCRI3_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235736/do79_CEBPA_liver_sc9314_cfa3_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235755/do376_FoxA1_liver_ab5089_rno5_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235729/do204_CEBPA_liver_sc9314_hsa1323_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235753/do732_HNF4a_liver_ARP31946_mmu0h490+491_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235778/do389_FoxA1_liver_ab5089_hsaHH1294_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235762/do474_FoxA1_liver_ab5089_mmlBlues_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235789/do558_HNF4a_liver_ARP31046_mmlBlues_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235734/do149_CEBPA_liver_sc9314_mmlBlues_CRI04.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235738/do255_FoxA1_liver_ab5089_cfa4_CRI02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235757/do762_HNF6_liver_sc13050_mmlBlues_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235773/do836_HNF6_liver_sc13050_mmlBob_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235730/do843_CEBPA_liver_sc9314_mmuBL60ON562_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235732/do79_CEBPA_liver_sc9314_cfa3_CRI02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235772/do76_HNF4a_liver_ARP31046_cfa4_SAN01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235771/do205_Input_liver_none_hsa1294_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235783/do132_Input_liver_none_mmlBlues_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235741/do566_Input_liver_none_mmuOON489_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235769/do149_CEBPA_liver_sc9314_mmlBlues_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235781/do466_FoxA1_liver_ab5089_mmuOON405_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235750/do133_Input_liver_none_mmlBlues_CRI02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235763/do562_HNF4a_liver_ARP31046_mmuOON489_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235742/do107_Input_liver_none_cfa4_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235774/do75_HNF4a_liver_ARP31046_cfa3_SAN02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235726/do61_CEBPA_liver_sc9314_cfa4_CRI03.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235731/do77_HNF6_liver_sc13050_cfa3_SAN03.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235777/do261_Input_liver_none_cfa3_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235740/do79_CEBPA_liver_sc9314_cfa3_CRI03.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235784/do851_Input_liver_none_mmuBL60ON562_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235780/do149_CEBPA_liver_sc9314_mmlBlues_CRI02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235723/do206_CEBPA_liver_sc9314_hsa1294_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235739/do377_CEBPA_liver_sc9314_rno5_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235722/do560_CEBPA_liver_sc9314_mmuOON489_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235748/do204_CEBPA_liver_sc9314_hsa1323_CRI02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235760/do902_HNF6_liver_sc13050_rno8_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235749/do764_HNF6_liver_sc13050_hsaHH1328_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235776/do118_CEBPA_liver_sc9314_mmlBlues_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235746/do61_CEBPA_liver_sc9314_cfa4_CRI02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235770/do811_Input_liver_none_rno8_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235752/do735_HNF6_liver_sc13050_mmu0h490+491_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235768/do507_HNF6_liver_sc13050_hsaHH1294_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235779/do149_CEBPA_liver_sc9314_mmlBlues_CRI03.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235747/do373_HNF6_liver_sc13050_rno5_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235767/do859_HNF4a_liver_ARP31046_rno7_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235725/do318_FoxA1_liver_ab5089_cfa3_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235743/do75_HNF4a_liver_ARP31046_cfa3_SAN03.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235765/do650_Input_liver_none_hsaHH1328_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235759/do205_Input_liver_none_hsa1294_CRI02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235758/do75_HNF4a_liver_ARP31046_cfa3_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235775/do61_CEBPA_liver_sc9314_cfa4_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235786/do463_FoxA1_liver_ab5089_mmuOON404_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235785/do855_Input_liver_none_mmlBob_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235724/do77_HNF6_liver_sc13050_cfa3_SAN04.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235728/do199_HNF4a_liver_ARP31046_hsaHH1308_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235766/do206_CEBPA_liver_sc9314_hsa1294_CRI02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235744/do810_Input_liver_none_rno7_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235790/do76_HNF4a_liver_ARP31046_cfa4_SAN02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235733/do901_FoxA1_liver_ab5089_rno8_CRI02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235764/do404_FoxA1_liver_ab5089_hsaCRI3_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235721/do133_Input_liver_none_mmlBlues_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235751/do100_Input_liver_none_cfa4_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235745/do77_HNF6_liver_sc13050_cfa3_CRI02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235735/do137_Input_liver_none_hsaHH1308_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235761/do203_Input_liver_none_hsa1323_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235727/do374_HNF4a_liver_ARP31046_rno5_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235756/do199_HNF4a_liver_ARP31046_hsaHH1308_CRI02.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235754/do505_HNF4a_liver_ARP31046_hsaHH1294_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235787/do860_CEBPA_liver_sc9314_rno7_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235782/do798_HNF6_liver_sc13050_mmu12_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235737/do255_FoxA1_liver_ab5089_cfa4_CRI01.fq.gz
wget -nc ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR235/ERR235720/do557_HNF4a_liver_ARP31046_mmlBlues_CRI01.fq.gz

# Gunzip all files
gunzip --keep *.fq.gz

# Decide whether to run fastqc or not!
if [ -d "${ROOT}/chip_seq_data/fastqc_results" ]; then
    printf "Using previous fastqc results!\n\n"
else 
    #  Generate fastqc files
    mkdir -p "${ROOT}/chip_seq_data/fastqc_results"
    fastqc -t 6 *.fq # use six threads
    mv *_fastqc.html *_fastqc.zip "${ROOT}/chip_seq_data/fastqc_results/"
fi