# Domain Adaptation-Enhanced Searchlight: Enabling brain decoding from visual perception to mental imagery

This repository contains the full code for the paper "Domain Adaptation-Enhanced Searchlight: Enabling brain decoding from visual perception to mental imagery" (Olza A., Soto D., Santana R., 2024).

## Repository structure :seedling:

The repository is organized as follows: 

- `algorithms/searchlight.py` contains the implementation of the DA-enhanced searchlight procedure.
- `DA_comparison` contains the comparison of DA techniques that motivates the use of the regular transfer method in the DA-enhanced searchlight, and the scripts for the ROI-based analysis.
- `DA_searchlight` contains the scripts for the experimental validation of the DA-enhanced searchlight.

## Dependencies :clipboard:
<details>

<summary>Click here to unfold the list of dependencies.</summary>

```bash
adapt
nilearn
matplotlib
seaborn
tqdm
pandas
pathlib
R: r-dplyr r-devtools r-tidyr
FSL
```
</details>

## Reproducibility :crystal_ball:
`reproduce.sh` reproduces all the results and figures in the paper with parameters `NITER=100`, `n_subj=18` and `n_reg=14`. 
Please note that the actual experimentation was heavily parallelized; running this script sequentially would take multiple days. 

The number of iterations for statistical validation can be reduced by changing the value of `NITER` to accelerate the script, without guarantee of obtaining the same statistical insights from the paper. 
With `NITER=1`, the total time is ... using 24 CPUs.

The fMRI data must be placed under a directory named `data` with the following structure, where `subject_id` goes from 1 to 18: 

<details>

<summary>Click here to unfold the required structure for the data directory.</summary>

```bash
data
├── imagery
│   ├── subject_id
│   │   ├── events.csv
│   │   ├── FFG.npy
│   │   ├── FP.npy
│   │   ├── IFGoperc.npy
│   │   ├── IFGorbital.npy
│   │   ├── IFGtriang.npy
│   │   ├── IPL.npy
│   │   ├── ITG.npy
│   │   ├── LOG.npy
│   │   ├── MOG.npy
│   │   ├── MTG.npy
│   │   ├── PCG.npy
│   │   ├── PCUN.npy
│   │   ├── SFG.npy
│   │   └── TP.npy
├── perception
│   ├── subject_id
│   │   ├── events.csv
│   │   ├── FFG.npy
│   │   ├── FP.npy
│   │   ├── IFGoperc.npy
│   │   ├── IFGorbital.npy
│   │   ├── IFGtriang.npy
│   │   ├── IPL.npy
│   │   ├── ITG.npy
│   │   ├── LOG.npy
│   │   ├── MOG.npy
│   │   ├── MTG.npy
│   │   ├── PCG.npy
│   │   ├── PCUN.npy
│   │   ├── SFG.npy
│   │   └── TP.npy
└── whole_brain
    ├── imagery
    │   ├── subject_id
    │   │   └── imagery_detrended_zscored_stacked_vols_of_interest_searchlight.nii.gz
    └── perception
        ├── subject_id
        │   ├── example_func_deoblique_brainmask.nii
        │   └── perception_detrended_zscored_stacked_vols_of_interest_searchlight.nii.gz
```
</details>
Files to transform the searchlight results to MNI space with FSL must be located in `DA_searchlight/searchlight_normalisation/participants`, with the following structure for each subject:

<details>

<summary>Click here to unfold the required structure for the transformation files.</summary>

```bash
participants
├── subject_id
│   └── perception
│       ├── preprocessed
│       │   └── mni
│       │       └── example_func
│       │           └── examplefunc2struct.mat
│       └── raw
│           └── anat
│               └── nonlinear_trans.nii.gz
```
</details>

