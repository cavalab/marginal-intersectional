# Intersectional Consequences for Marginal Fairness in a Prediction Model for Emergency Room Admissions

## Authors

Elle Lett, Shakiba Shahbandegan, Yuval Barak-Corren, Andrew M. Fine, Ben Y. Reis,  William G. La Cava
 
## Abstract

This study addresses the challenge of achieving fair treatment in emergency room (ER) admissions, considering the prevalent health disparities among marginalized populations, especially minoritized ethnoracial groups. 
These populations often face extended wait times and adverse health outcomes in ERs. 
We propose fairness-aware machine learning models to mitigate racialized health inequities by focusing on intersectional fairness, which considers multiple demographic traits together, unlike traditional "marginal" fairness that examines attributes in isolation. 

## Methods

## Datasets

To use the MIMIC-IV admissions dataset, you must first access the data from https://physionet.org/content/mimiciv/1.0/ and https://physionet.org/content/mimic-iv-ed/. 
See https://github.com/cavalab/mimic-iv-admissions for our pre-processing scripts.

Boston Children's Hospital (BCH) data is not publicly available. However the preprocessing scripts are viewable in `clean_BCH.py`. 


## Install

Clone this repository:

```
git clone https://github.com/cavalab/marginal_intersectional
```

`environment.yml` includes the conda environment specification for the experiments. 
Use conda or mamba to install it, e.g. from the repo folder run

```
conda env create
```


### Using MCBoost to optimize multi-calibration

See the script `run_multicalibration_experiment.py`. 
To run the multicalibration experiment on MIMIC-IV, run

```
python run_multicalibration_experiment.py mimic result-directory
```

### Using FOMO for controlling subgroup false negative rates and balanced accuracy

The script `single_fomo_experiment.py` will run a single training instance of FOMO on a given dataset under a specific scenario. 
For example: 

```
python single_fomo_experiment.py -base_est lr -metric FNR -scenario Marginal -gamma True -problem linear -seed 42 -rdir results_fomo_2023-08-07
```

If you are using a SLURM cluster, you can use `submit_fomo_experiment.py` to generate job scripts and submit multiple experiments simultaneously. 
`run_fomo_experiment.sh` shows an example call to `submit_fomo_experiment.py`. 
