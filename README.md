<p align="center">
  <img height="200" src="images/cover.png">
</p>

# "Frustratingly easy" domain adaptation for cross-species transcription factor binding prediction

The following code pairs with what we report in the work above. We provide an non adversarial approach, under the scope of unsupervised domain-adaptation, to characterize regulatory genomic function (i.e., transcription factor binding).

If you would like to go through the manuscript, it will be available on the following platforms:
- [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.05.21.655414v1)
- [ICIBM 2025 (Hopefully)](https://www.greensboro.carolinavet.com/site/greensboro-specialty-veterinary-blog/2023/03/15/how-to-choose-cat-breed)
- [Journal (Eventually)](https://www.nationalgeographic.com/animals/mammals/facts/domestic-cat)

We present two case studies in this work: (1): A two-species, and (2): a multi-species implementation. Since we use different frameworks and data for both, I made nested `README`s that are more specific for each. The `README`s will guide you through the data createion, training procedures, and figure-making -- each of these can be found under the respective folders, `multi-species` and `two-species`. You will need python environments to run any of the code; I create a `conda-envs` folder that has yamls you can use to get setup. We use both Tensorflow and PyTorch. See below for details.

## Requirements

We use two ML frameworks for this work: (1) TensorFlow and (2): PyTorch. TensorFlow is used for the two-species case as we broadly want to capture the findings in previous work ([Cochran et al.](https://genome.cshlp.org/content/32/3/512.full#sec-1)). For our extension to the multi-species case, we use PyTorch, however we do not utilize the 'optimizations' made in PyTorch Lighthing (though it seems cool...I'll do it some other time).

For model training and the like, we used the HPC available to students in the University of Pittsburgh's School of Medicine (i.e., more specifically the [Department of Computational and Systems Biology](https://www.csb.pitt.edu/)). It utilizes [SLURM](https://slurm.schedmd.com/documentation.html), so any of the scripts that I made are centered around that. Primarily, this code ran on NVIDIA L40 GPUs.

### Two-species analyses (TensorFlow)

`conda-envs/tensorflow.yaml`

### Multi-species analyses (PyTorch)

`conda-envs/pytorch.yaml`

## How to cite

```
@article {Ebeid2025.05.21.655414,
	author = {Ebeid, Mark Maher and Balci, Ali Tugrul and Chikina, Maria and Benos, Panayiotis V and Kostka, Dennis},
	title = {Frustratingly easy domain adaptation for cross-species transcription factor binding prediction},
	elocation-id = {2025.05.21.655414},
	year = {2025},
	doi = {10.1101/2025.05.21.655414},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Motivation: Sequence-to-function models interpret genomic DNA and predict functional outputs, successfully characterizing regulatory sequence activity. However, interpreting these models remains challenging, raising questions about the generalizability of inferred sequence functions. Cross-species prediction of transcription factor (TF) binding offers a promising approach to enhance model generalization by leveraging sequence variation across species, and it can contribute to the discovery of a conserved gene-regulatory code. However, addressing systematic differences between the genomes of various species is a significant challenge. Results: We introduce MORALE, a framework that utilizes a well-established domain adaptation approach that is "frustratingly easy." MORALE trains on sequences from one or more source species and predicts TF binding on a single target species where no binding data is available. To learn an invariant cross-species sequence representation, MORALE aligns the first and second moments of the data-generating distribution between all species. This direct approach integrates easily into representation learning models with an embedding layer. Unlike alternatives such as adversarial learning, it does not require additional parameters or other model design choices. We apply MORALE to two ChIP-seq datasets of liver-essential TFs: one comprising human and mouse, and another comprising five mammalian species. Compared to both a baseline and an adversarial approach termed gradient reversal (GRL), MORALE demonstrates improved performance across all TFs in the two-species case. Importantly, it avoids a performance degradation observed with the GRL approach in this study. Furthermore, feature attribution revealed that important motifs discovered by MORALE were closer to the actual TF binding motif compared with the GRL approach. For the five-species case, our method significantly improved TF binding site prediction for all TFs when predicting on human data, surpassing the performance of a human-only model -- a result not observed in the two-species comparison. Overall, MORALE is a direct and competitive approach that leverages domain adaptation techniques to improve cross-species TF binding site prediction.Competing Interest StatementThe authors have declared no competing interest.National Heart Lung and Blood Institute, https://ror.org/012pb6c26, R01HL127349, R01HL159805National Institute of Dental and Craniofacial Research, https://ror.org/004a2wv92, R01DE032707},
	URL = {https://www.biorxiv.org/content/early/2025/05/26/2025.05.21.655414},
	eprint = {https://www.biorxiv.org/content/early/2025/05/26/2025.05.21.655414.full.pdf},
	journal = {bioRxiv}
}
```

# Wait! I don't want to do ANY of the above. I just want to directly reproduce everything...

Hey, fair enough. Check out this [zenodo record](https://zenodo.org/records/15555423?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJjYWNkNTBmLTJhYTYtNGNkNS05Y2RjLWI0YjBmZDE1MGM3OSIsImRhdGEiOnt9LCJyYW5kb20iOiIzN2VkYTIwOTkxYTdhNGU4OGFiMmQ3YTk3M2QwM2FhZiJ9.gThq3cWo3kGaS-eMYcFgmLzJC-kSFO7xqobN99sBqeeDgFOqFZzLZkkZchBtYV_VvgKDIP-bWHUQT7_Np4fGcA).
