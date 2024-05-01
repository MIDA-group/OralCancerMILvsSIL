# Deep multiple instance learning versus conventional deep single instance learning for interpretable oral cancer detection
<a href="mailto:nadezhda.koriakina@it.uu.se">Nadezhda Koriakina</a>:envelope:, <a href="mailto:natasa.sladoje@it.uu.se">Nataša Sladoje</a>, Vladimir Bašić and <a href="mailto:joakim.lindblad@it.uu.se">Joakim Lindblad</a>

## Table of Contents
1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [How to use](#how-to-use)
4. [Citation](#citation)
5. [References](#references)
6. [Acknowledgements](#acknowledgements)

## Overview
- Code for creating PAP-QMNIST-bags datasets
- Code for training and evaluating Attention-based deep multiple instance learning (ABMIL)[[1]](#1) with within bag sampling [[2]](#2) for PAP-QMNIST-bags and oral cancer (OC) datasets
- Code for training and evaluating single instance learning (SIL) for PAP-QMNIST-bags and OC datasets

Based on [original implementation](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) of ABMIL and [modified implementation](https://github.com/MIDA-group/SampledABMIL) with within bag sampling

## Dependencies
Python version 3.8.8. Main dependencies: reqs.txt

## How to use
- `Create_PAP_QMNISTbags_datasets.ipynb`: code for creating PAP-QMNIST-bags datasets. QMNIST [[3]](#3) images are colorised according to the distribution of color channels corresponding to OC sample, augmented, resized to the size of images from OC dataset. The bags are created with the same number of instances as in OC dataset.
- directory `MIL`: codes for training and evaluating ABMIL with within bag sampling on PAP-QMNIST-bags and OC datasets.
- directory `SIL`: codes for training and evaluating conventional deep SIL on PAP-QMNIST-bags and OC datasets.

Restart jupyter kernel if changes to the internal codes are made.

<ins>Note:</ins> the code is created for PAP-QMNIST data based on OC data and might require changes if custom data is used.

## Citation
@article{10.1371/journal.pone.0302169,
    doi = {10.1371/journal.pone.0302169},
    author = {Koriakina, Nadezhda AND Sladoje, Nataša AND Bašić, Vladimir AND Lindblad, Joakim},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Deep multiple instance learning versus conventional deep single instance learning for interpretable oral cancer detection},
    year = {2024},
    month = {04},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pone.0302169},
    pages = {1-23},
    abstract = {The current medical standard for setting an oral cancer (OC) diagnosis is histological examination of a tissue sample taken from the oral cavity. This process is time-consuming and more invasive than an alternative approach of acquiring a brush sample followed by cytological analysis. Using a microscope, skilled cytotechnologists are able to detect changes due to malignancy; however, introducing this approach into clinical routine is associated with challenges such as a lack of resources and experts. To design a trustworthy OC detection system that can assist cytotechnologists, we are interested in deep learning based methods that can reliably detect cancer, given only per-patient labels (thereby minimizing annotation bias), and also provide information regarding which cells are most relevant for the diagnosis (thereby enabling supervision and understanding). In this study, we perform a comparison of two approaches suitable for OC detection and interpretation: (i) conventional single instance learning (SIL) approach and (ii) a modern multiple instance learning (MIL) method. To facilitate systematic evaluation of the considered approaches, we, in addition to a real OC dataset with patient-level ground truth annotations, also introduce a synthetic dataset—PAP-QMNIST. This dataset shares several properties of OC data, such as image size and large and varied number of instances per bag, and may therefore act as a proxy model of a real OC dataset, while, in contrast to OC data, it offers reliable per-instance ground truth, as defined by design. PAP-QMNIST has the additional advantage of being visually interpretable for non-experts, which simplifies analysis of the behavior of methods. For both OC and PAP-QMNIST data, we evaluate performance of the methods utilizing three different neural network architectures. Our study indicates, somewhat surprisingly, that on both synthetic and real data, the performance of the SIL approach is better or equal to the performance of the MIL approach. Visual examination by cytotechnologist indicates that the methods manage to identify cells which deviate from normality, including malignant cells as well as those suspicious for dysplasia. We share the code as open source.},
    number = {4},
}

## References
<a id="1">[1]</a> 
M.  Ilse,  J.  Tomczak,  and  M.  Welling,  “Attention-based  deep  multiple instance learning,”  in International conference on machine learning.PMLR, 2018, pp. 2127–2136.<br />
<a id="2">[2]</a> 
N. Koriakina, N. Sladoje and J. Lindblad, "The Effect of Within-Bag Sampling on End-to-End Multiple Instance Learning," 2021 12th International Symposium on Image and Signal Processing and Analysis (ISPA), 2021, pp. 183-188, doi: 10.1109/ISPA52656.2021.9552170.<br />
<a id="1">[3]</a> 
Yadav, Chhavi, and Léon Bottou. "Cold case: The lost mnist digits." arXiv preprint arXiv:1905.10498 (2019).<br />

## Acknowledgements
This work is supported by: Sweden’s Innovation Agency (VINNOVA), grants 2017-02447, (J.L.), 2021-01420 (J.L.), and 2020-03611 (J.L.), the Swedish Research Council, grant 2017-04385 (J.L.) and 2022-03580_VR (N.S.), and Cancerfonden, project number 22 2353 Pj (J.L.) and project number 22 2357 Pj (N.S.).
A part of the experiments was enabled by computational resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725.




 
