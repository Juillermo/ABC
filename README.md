# Associations Beyond Chance (ABC)

The ABC model provides a **Bayesian framework to infer multimorbidity associations between health conditions** from Electronic Health Records. The outputs are posterior distribution over pairwise association values, which can be assembled and visualised as multimorbidity weigthed network.

The ABC model was presented on the article [*"Multimorbidity analysis with low condition counts: a robust Bayesian approach for small but important subgroups"*](https://authors.elsevier.com/sd/article/S2352396424001166). This repository also provides code to reproduce the experiments and visualisations there. If you use this code, please cite the paper:

> Romero Moreno G., Restocchi V., Fleuriot JD., Anand A., Mercer SW., Guthrie B. (2024). [*Multimorbidity analysis with low condition counts: a robust Bayesian approach for small but important subgroups*](https://www.sciencedirect.com/science/article/pii/S2352396424001166); **eBioMedicine**, Volume 102, 105081, ISSN 2352-3964, https://doi.org/10.1016/j.ebiom.2024.105081.

```
@article{ROMEROMORENO2024105081,
title = {Multimorbidity analysis with low condition counts: a robust Bayesian approach for small but important subgroups},
journal = {eBioMedicine},
volume = {102},
pages = {105081},
year = {2024},
issn = {2352-3964},
doi = {https://doi.org/10.1016/j.ebiom.2024.105081},
url = {https://www.sciencedirect.com/science/article/pii/S2352396424001166},
author = {Guillermo {Romero Moreno} and Valerio Restocchi and Jacques D. Fleuriot and Atul Anand and Stewart W. Mercer and Bruce Guthrie},
}
```

![](1-s2.0-S2352396424001166-gr4_lrg.jpg)


## Installation and dependencies

You can install and use the package just by running `python setup.py install`. (It is recommended to perform the installation in a python virtual environment.)

Or alternatively:

- Install [Anaconda](https://docs.anaconda.com/) (if not already installed)  
- Execute `conda env create -n ABC --file packages.yml`* in a terminal for creating an environment called `ABC` with all the required packages. *Be aware that it may take a few GB of space.*
- Activate the environment with `conda activate ABC`, and run the code or set up a jupyter notebook server (by running `jupyter notebook`)

> .* Or you can directly execute `conda create -n ABC -c conda-forge numpy=1.22.3 scipy=1.8.0 pandas=1.4.2 matplotlib=3.5.1 seaborn=0.13.1 networkx=2.8 bokeh=3.3.0 cmdstanpy=1.1.0 jupyter`.

While the code is in *python*, Bayesian inference is performed via [Stan](http://mc-stan.org) through the package `cmdstanpy` (version 1.1.0), providing a python API to the *Stan* library. The model (defined in the file [`models/MLTC_atomic_hyp_mult.stan`](models/MLTC_atomic_hyp_mult.stan)) could also work with [any stan interface](https://mc-stan.org/users/interfaces/index.html).

## Using the model

Our model can be used simply by running `ABC "path/to/dataset_file.csv"`, which will fit the model and generate output files with the results. For more information on additional argumnets, run `ABC --help`.

Additionally, you can integrate our model into other *python* code directly. You can see an example snippet on how to do so below.

```python
from ABC.model import ABCModel
from ucimlrepo import fetch_ucirepo 

# This loads an example dataset. Swap these lines for those loading your dataset
dataset = fetch_ucirepo(name='CDC Diabetes Health Indicators')
data = pd.concat([dataset.data.features, dataset.data.targets ], axis=1)
bin_vars = dataset.variables[dataset.variables["type"] == "Binary"]["name"].to_list()
# Make sure to use columns with binary variables only

model = ABCModel()
model.load_fit(data, "choose_name_for_model", column_names=bin_vars, num_warmup=500, num_samples=2000, random_seed=1)

ABC = model.get_associations()  # This retrieves the whole distribution for all association pairs
results = model.get_results_dataframe(credible_inteval_pvalue=0.01)  # This creates a table with summary statistics
```

A detailed example with **step-by-step intructions** on how to use the model and produce outputs and visualisations within python code can be found at the tutorial notebook ['ABC_to_ABC.ipynb'](notebooks/ABC_to_ABC.ipynb).


## Reproducing results

You can replicate the results and figures from the [*"Multimorbidity analysis with low condition counts: a robust Bayesian approach for small but important subgroups"*](https://authors.elsevier.com/sd/article/S2352396424001166) article by running the notebook [`notebooks/results.ipynb`](notebooks/results.ipynb). However, note that this will only be possible if you have access to the dataset used in the study.

You can still reproduce the results shown in that notebook on a different dataset, for which you will need to adapt all functions and variables within the file [`ABC/data.py`](ABC/data.py) to your dataset characteristics and then rerun [`notebooks/results.ipynb`](notebooks/results.ipynb) --- or use the functions in the file [`ABC/results.py`](ABC/results.py).


## Repository structure

* [`ABC/`](ABC/): python files with the basic classes and functions.
* [`models/`](models/): files defining *Stan* models.
* [`output/`](output/): folder in which to save the fitted models.
* [`notebooks/`](notebooks/): results and examples implementing our models and code.
* [`figs/`](figs/): folder in which to save the figures produced in the notebooks.


## Acknowledgements

Functions and notebooks were inspired by [this repository](https://github.com/jg-you/plant-pollinator-inference/tree/master).


### Contact

Any question, comment, or feedback, contact <Guillermo.RomeroMoreno@ed.ac.uk>, or submit an Issues on GitHub.
