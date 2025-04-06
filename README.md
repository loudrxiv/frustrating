<p align="center">
  <img height="200" src="images/cover.png">
</p>

# "Frustratingly easy" domain adaptation for cross-species transcription factor binding prediction

The following code pairs with the analyses done in the work mentioned above. We provide an alternative approach, under the scope of unsupervised domain-adaptation, to successfully characterize regulatory genomic function (i.e., transcription factor binding).

If you would like to go through the manuscript first, it is not yet publicly available, but will be available on the following platforms:
- [bioRxiv](https://www.britannica.com/animal/cat)
- [Journal](https://www.nationalgeographic.com/animals/mammals/facts/domestic-cat)

For each case in the paper: (1): Two-species, or (2): Multi-species, we offer a tailored `readme` to guide you through our training procedures -- these are detailed under the `src\X` folders. Below is a more general guide of how to set up the overall environments needed.

## Requirements

We use two ML frameworks for this work: (1) TensorFlow and (2): PyTorch. TensorFlow is used for the two-species case as we broadly want to capture the findings in previous work ([Cochran et al.](https://genome.cshlp.org/content/32/3/512.full#sec-1)). For our extension to the multi-species case, we use PyTorch, however we do not utilize the 'optimizations' made in PyTorch Lighthing (though it seems cool...).

For model training and the like, we used the HPC (it utilizes SLURM) available to us through the University of Pittsburgh's CSB department (part of the school of medicine). Primarily, our code ran on NVIDIA L40 GPUs.

### Two-species analyses (TensorFlow)

We offer a exported `yaml` via conda that we used for training: `conda-envs/tensorflow.yaml`

### Multi-species analyses (PyTorch)

We offer a exported `yaml` via conda that we used for training: `conda-envs/pytorch.yaml`

## How to cite

```
TBD
```
