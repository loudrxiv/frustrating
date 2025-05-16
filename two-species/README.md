## Setup

Here we step through each phase of the paper setup in order to reproduce findings in the work --- from data creation to model training for the two-species analysis. The first thing to do is to run `setup_directories_and_download_files.sh`. I have hard-coded the paths for this file, which you *need* to edit. So go into the file, change the `ROOT` to be the path to the `two-species` directory. Afterwards, just run the 1<sup>st</sup> script like so:

```
./setup_directories_and_download_files.sh
```

Afterwards we are ready to proceed, stepwise, to: (1) create the data, (2) run the models, and (3) perform downstream analyses.

### 0. Preprocess the genome and the peaks

This first step focuses on gathering files and pretty much 'tiling' the genome based on the window size we want to operate off of. We download the necessary files for hg38 and mm10, filter it out (in bash!) using the blacklist files and then tile the genomes into 500bp with 50bp offset.

To run all steps needed, simply execute:
```
./0_runall_preprocess_data.sh
```

However, BEFORE running the preprocessing steps, you must have data. Please refer to the supplementary material (ST7, I think) in the manusscript to find all avaialble accession IDs needed to download the raw data. The next step is to call peaks with an algorithm of your choosing. We follow the previous procedure in relying on [multiGPS](https://github.com/seqcode/multigps) in order to do this; the scripts can be found in the `peak_calling` folder. We mimic their nested folder structure to organize the BED files.

```
└── raw_data
    ├── hg38
    │   ├── CEBPA
    │   │   └── mgps_out_CEBPA.bed
    │   ├── CTCF
    │   │   └── mgps_out_CTCF.bed
    │   ├── Hnf4a
    │   │   └── mgps_out_Hnf4a.bed
    │   └── RXRA
    │       └── mgps_out_RXRA.bed
    └── mm10
        ├── CEBPA
        │   └── mgps_out_CEBPA.bed
        ├── CTCF
        │   └── mgps_out_CTCF.bed
        ├── Hnf4a
        │   └── mgps_out_Hnf4a.bed
        └── RXRA
            └── mgps_out_RXRA.bed
```

If you want to change this, use a different software, or whatever else: you **MUST** alter the code 👿.

### 1. Make training and testing data

Here we do a couple things, we follow previous work in creating specialized data from 15 epochs of training, and holdout chromosomes 1 & 2 for validation and testing. We create a few more files for repeat annotation analyses where applicable. You have to change the hard-coded `ROOT` here too. Afterwards, simply run:
```
./1_runall_setup_model_data.sh
```

### 2. Tune, train, and test models

We provide all code used to tune, train and test models in this directory. In order to run these files, we used NVIDIA L40s. Specalized components for the Gradient Reversal & Moment Alignment models are provided in their subdirectories.

### 3. Generate manuscript figures and tables

We provide all code used to generate manuscript figures here. There are in form (mainly) of ipynbs.