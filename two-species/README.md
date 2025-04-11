## Setup

Here we step through each phase of data creation and model training for the two-species analysis. The first thing to do is to run `setup_directories_and_download_files.sh`, which asks of the user for the root of the project. The root will contain all the relevant data and directories for saving the output. Run it like so:

```
./setup_directories_and_download_files.sh $ROOT
```

Afterwards we are ready to proceed, stepwise, to: (1) create the data, (2) run the models, and (3) perform downstream analyses.

### 0. Preprocess the genome and the peaks

This first step focuses on gathering files and pretty much 'tiling' the genome based on the window size we want to operate off of. We download the necessary files for hg38 and mm10, filter it out (in bash!) using the blacklist files and then tile the genomes into 500bp with 50bp offset.

To run all steps needed, simply execute:

```
./runall_preprocess_data.sh
```

However, BEFORE running the preprocessing steps, you must have data. Please refer to the [supplementary material](https://genome.cshlp.org/content/suppl/2022/02/14/gr.275394.121.DC1/Supplemental_Table_S5.pdf) available from previous work to obtain all needed codes to download the raw data. The next step is to call peaks with an algorithm of your choosing. We follow the previous procedure in relying on [multiGPS](https://github.com/seqcode/multigps) in order to do this; the scripts can be found in the `peak_calling` folder. We mimic their nested folder structure to organize the BED files.

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

If you want to change this, you MUST alter the code 👿.

### 1. Make training and testing data

Here we do a couple things, we follow previous work in creating specialized data from 15 epochs of training, and holdout chromosomes 1 & 2 for validation and testing. We create a few more files for repeat annotation analyses where applicable. Simply run

```
./runall_setup_model_data.sh
```

### 2. Tune, train, and test models

We provide all code used to tune, train and test models in this directory. In order to run these files, we used NVIDIA L40s. Specalized components for the Gradient Reversal & Moment Alignment models are provided in their subdirectories.

### 3. Generate manuscript figures and tables

We provide all code used to generate manuscript figures here. There are in form (mainly) of ipynbs.

### Zenodo

We offer the peak calls used here in [this]() record.
