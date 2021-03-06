# Oral cancer detection and interpretation: Deep multiple instance learning versus conventional deep single instance learning
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
@article{koriakina2022oral,<br />
  title={Oral cancer detection and interpretation: Deep multiple instance learning versus conventional deep single instance learning},<br />
  author={Koriakina, Nadezhda and Sladoje, Nata{\v{s}}a and Ba{\v{s}}i{\'c}, Vladimir and Lindblad, Joakim},<br />
  journal={arXiv preprint arXiv:2202.01783},<br />
  year={2022}<br />
}

## References
<a id="1">[1]</a> 
M.  Ilse,  J.  Tomczak,  and  M.  Welling,  “Attention-based  deep  multiple instance learning,”  in International conference on machine learning.PMLR, 2018, pp. 2127–2136.<br />
<a id="2">[2]</a> 
N. Koriakina, N. Sladoje and J. Lindblad, "The Effect of Within-Bag Sampling on End-to-End Multiple Instance Learning," 2021 12th International Symposium on Image and Signal Processing and Analysis (ISPA), 2021, pp. 183-188, doi: 10.1109/ISPA52656.2021.9552170.<br />
<a id="1">[3]</a> 
Yadav, Chhavi, and Léon Bottou. "Cold case: The lost mnist digits." arXiv preprint arXiv:1905.10498 (2019).<br />

## Acknowledgements
This work is supported by: Sweden’s Innovation Agency (VINNOVA), grants 2017-02447 and 2020-03611, and the SwedishResearch Council, grant 2017-04385. A part of computations was enabled by resources provided by the Swedish NationalInfrastructure for Computing (SNIC) at Chalmers Centre for Computational Science and Engineering (C3SE), partially fundedby the Swedish Research Council through grant no. 2018-05973.



 
