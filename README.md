# distribution-optimization-py: Optimization-Based GMM Parameter Estimation

This repository contains code and scripts for **Gaussian Mixture Model (GMM) parameter estimation** using optimization methods.
- Generate synthetic datasets
- Benchmark optimization algorithms
- Reproduce plots and figures presented in the paper

## Table of Contents
1. [Installation](#installation)
2. [Generating Datasets](#generating-datasets)
3. [Benchmarking Optimization Algorithms](#benchmarking-optimization-algorithms)
4. [Reproducing Plots](#reproducing-plots)

---

## Installation

1. Clone the repository:
   ```bash
   git clone git+https://github.com/agh-a2s/distribution-optimization.git
   ```
2. Navigate to the cloned directory:
   ```bash
   cd distribution-optimization
   ```
3. Install dependencies (using [Poetry](https://python-poetry.org/)):
   ```bash
   poetry install
   ```

---

## Generating Datasets

Synthetic datasets used in the paper are stored in the `results` directory.  
To regenerate these datasets, run:

```bash
poetry run python3 -m distribution_optimization_py.experiment
```

This script will produce or update synthetic datasets inside the `results` folder.

---

## Benchmarking Optimization Algorithms

To benchmark different optimization algorithms for GMM parameter estimation, run:

```bash
poetry run python3 -m distribution_optimization_py.solver
```

---

## Reproducing Plots

All plots are generated and saved to the `images` directory. The scripts below reproduce the figures from the paper:

- **Fig. 1:**  
  ```bash
  poetry run python3 -m distribution_optimization_py.plot_binning_scheme_difference
  ```

- **Fig. 2:**  
  ```bash
  poetry run python3 -m distribution_optimization_py.experiment.plot
  ```

- **Fig. 3:**  
  ```bash
  poetry run python3 -m distribution_optimization_py.solver.plot
  ```

- **Fig. 4:**  
  ```bash
  poetry run python3 -m distribution_optimization_py.compare_results
  ```
