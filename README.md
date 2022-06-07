# Code release for WALNUT

## Overview

This repository contains the baseline code for the WALNUT paper published in NAACL 2022. Detailed description about the data sets and methods can be manuscript at [here](https://arxiv.org/pdf/2108.12603.pdf).

## Getting data

Data for WALNUT can be downloaded from [here](https://github.com/gkaramanolakis/WALNUT_data).

## Repo structure

`document-level-baselines` contains source codes for 5 baseline mthods (C, W, Snorkel, C+W, C+Sonrkel) for document level classification tasks;

`document-level-GLC_MWNET_MLC` contains source codes for 3 advanced semi-weakly sueprvised learning methods (GLC, MetaWN, MLC) for document-level classification tasks;

`token-level-baselines` contains source codes for 5 baseline mthods (C, W, Snorkel, C+W, C+Sonrkel) for token-level classification tasks;

`token-level-GLC_MWNET_MLC` contains source codes for 3 advanced semi-weakly sueprvised learning methods (GLC, MetaWN, MLC) for token-level classification tasks.

## Citation

If you find WALNUT useful, please cite the following paper

```
@inproceedings{zheng2022walnut,
  title={WALNUT: A Benchmark on Semi-weakly Supervised Learning for Natural Language Understanding},
  author={Guoqing Zheng, Giannis Karamanolakis, Kai Shu, Ahmed Hassan Awadallah},
  booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2022}
}
```

This code repository is released under MIT License. (See [LICENSE](LICENSE))
