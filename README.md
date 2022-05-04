# Code and Data for WALNUT

## Overview

This repo contains data and code for all 8 methods tested on 8 datasets in WALNUT paper. Detailed description about the data sets and methods can be manuscript.

## Repo structure

`data` contains all eight data sets with the 5 splits of clean/weak train (in readable JSON formats);

`document-level-baselines` contains source codes for 5 baseline mthods (C, W, Snorkel, C+W, C+Sonrkel) for document level classification tasks;

`document-level-GLC_MWNET_MLC` contains source codes for 3 advanced semi-weakly sueprvised learning methods (GLC, MetaWN, MLC) for document-level classification tasks;

`token-level-baselines` contains source codes for 5 baseline mthods (C, W, Snorkel, C+W, C+Sonrkel) for token-level classification tasks;

`token-level-GLC_MWNET_MLC` contains source codes for 3 advanced semi-weakly sueprvised learning methods (GLC, MetaWN, MLC) for token-level classification tasks.
