# distribution-optimization-py: Optimization-Based GMM Parameter Estimation

## Installation

```bash
git clone git+https://github.com/agh-a2s/distribution-optimization.git
```

To install dependencies
```
poetry install
```

## Reproducing Experiments and Plots

### Generating Datasets

Synthetic datasets are available in `results` directory. The generation can be reproduced by running `poetry run python3 -m distribution_optimization_py.experiment`.

### Benchmarking Optimization Algorithms

To benchmark optimization algorithms on GMM parameter estimation run `poetry run python3 -m distribution_optimization_py.solver`.

### Reproducing Plots from the Paper

Fig. 1:
```bash
poetry run python3 -m distribution_optimization_py.plot_binning_scheme_difference
```
Fig. 2:
```bash
poetry run python3 -m distribution_optimization_py.experiment.plot
```

Fig. 3:
```bash
poetry run python3 -m distribution_optimization_py.solver.plot
```

Fig. 3:
```bash
poetry run python3 -m distribution_optimization_py.compare_results
```

All these scripts will create plots in the `images` directory.

