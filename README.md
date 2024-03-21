# Associations Beyond Chance

This repository provides code linked to the paper [*"Multimorbidity analysis with low condition counts: a robust Bayesian approach for small but important subgroups"*](https://authors.elsevier.com/sd/article/S2352396424001166).

While the code is in *python*, Bayesian inference is performed via [Stan](http://mc-stan.org) through the package `cmdstanpy`, providing a python API to the *Stan* library.

![](1-s2.0-S2352396424001166-gr4_lrg.jpg)


## Packages

- numpy
- scipy
- pandas
- networkx
- cmdstanpy (pystan)
- matplotlib
- seaborn
- bokeh


## Using the model

Our model can be used via the class `ABCModel` within the `lib/model.py` file. Fitting the model and obtaining results can be done simply with

```python
from lib.model import ABCModel

model = ABCModel()
model.load_fit(data, "a_name_for_the_saved_model", num_warmup=500, random_seed=1)
results = model.get_results_dataframe(pvalue=0.01)
```

where `data` is a `pandas.DataFrame` object containing your dataset in **long format** --- i.e. with patients as rows, columns as variables, and binary values (diagnosis present / absent).


## Reproducing results

If you want to reproduce the results and figures from the article, you will first need to go to `lib/data.py` and check and modify all functions and variables within that file to adapt them to your dataset characteristics.
Then, you can rerun the cells within the jupyter notebook `notebooks/results.ipynb`, or use the functions in `lib/results.py`.


## Acknowledgements

Functions and notebooks were inspired by [this repository].
