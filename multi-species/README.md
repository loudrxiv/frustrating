## Setup

This is the subdirectory for the multi-species analyses, where we will step through each broad phase of data creation and model training. The first thing you must do is setup the directories in a similar manner to how the scripts expect:

```
./setup_directories.sh
```

which ends up calling a subscript (`0-1_ena-file-download-read_run-PRJEB1571-submitted_ftp-20240816-1712.sh`) which gets all the data we need from the mutli-species, ChIP-Seq data mentioned and created in [this previous work](https://elifesciences.org/articles/02626). Afterwards, we are ready to step through each phase!

## 1. Downloading, aligning, and calling the data + creating the datasets

All of these aforementioned steps are controlled by the master script: `run_pipeline.sh`. The way this script works is that there are several steps you need to step through in order to get ML ready data for model training, the enumerated steps are as follows:

1. `1_generate_data` calls `genomepy` to get the relevant, genome files we use to further downstream data creation, afterwards, we tile each genome based on a specified window size.
2. `2_align_to_reference` then aligns the relevant chip files to each of the references (we do little alteration to defaults here)
3. `3_call_peaks` calls `multiGPS` in order to call the peaks across controls and experiements from the data we pulled from
4. And finally, `4_create_all-alls` generates binary labels and such for those windows

You must run each step. This is how you can do it:

```
./run_pipeline.sh 1_generate_data
```

 ## 2. Splitting created datasets into training, validation, and testing

Now that we have the data, all together, our next step is to split it into training, testing and validation! Once again, we have a master script that controls these processes. We solve a linear program in order to etch ~10% of data for validation and ~10% for testing, with the idea being we preserve the overall binding ratio. In order to split the data, run:

```
./create_all_data.sh
```

## 3. Train and test models

We provide tuning, training, and testing code for all models here.

## 4. Creating figures

We also include the notebooks that we used to generate each of the figures in the main manuscript.
