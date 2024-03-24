# distribution-optimization-py: Distribution Optimization in Python

Install required dependencies:
```bash
poetry install --with dev
```

To reproduce ELA classification results use: `experiments/ela_analysis.ipynb`.

To reproduce datasets used for experiments use `experiments/create_datasets.ipynb`.

Hyperparameter optimization scripts:

* `scripts/optuna_cma_es_bipop_chromatogram_time.py`
* `scripts/optuna_de_ilshade_chromatogram_time.py`
* `scripts/optuna_hms_chromatogram_time.py`
* `R/optimize_hyperparams.R`

Notebooks (+ script) employed to save solutions:

* `experiments/cma_es_bipop.ipynb`
* `experiments/ilshade.ipynb`
* `experiments/hms_scaled.ipynb`
* `R/distribution_optimization.R`

Tables, boxplots were generated using `R/distribution_optimization.R`.

The source code for DistributionOptimization can be found in `R/DistributionOptimization`. It was adjusted with objective function wrapper -- to make sure that the budget is not exceeded.
