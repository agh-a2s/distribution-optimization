import os
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from distribution_optimization_py.problem import GaussianMixtureProblem
from distribution_optimization_py.solver import GASolver, Solver, iLSHADESolver
from distribution_optimization_py.solver.em import EMSolver
from distribution_optimization_py.synthetic_datasets import (
    generate_difficult_mixture_parameters,
    generate_mixture_parameters,
)
from distribution_optimization_py.utils import js_div, kl_div

RESULTS_DIR_NAME = "results"
MIN_OVERLAP_VALUE = 0.0
MAX_OVERLAP_VALUE = 0.5


@dataclass
class Method:
    solver: Solver
    bin_type: str
    bin_number_method: str
    name: str
    use_correction: bool = False


METHODS = [
    Method(
        solver=GASolver(),
        bin_type="equal_width",
        bin_number_method="keating",
        use_correction=True,
        name="DO",
    ),
    Method(
        solver=GASolver(),
        bin_type="equal_probability",
        bin_number_method="mann_wald",
        use_correction=True,
        name="GA Mann-Wald",
    ),
    Method(
        solver=iLSHADESolver(),
        bin_type="equal_probability",
        bin_number_method="mann_wald",
        use_correction=True,
        name="iLSHADE Mann-Wald",
    ),
    Method(
        solver=EMSolver(),
        bin_type="equal_probability",
        bin_number_method="mann_wald",
        use_correction=True,
        name="EM",
    ),
]


def run_experiment_for_nr_of_components(
    nr_of_components: int,
    nr_of_datasets: int,
    min_overlap_value: float = MIN_OVERLAP_VALUE,
    max_overlap_value: float = MAX_OVERLAP_VALUE,
    results_dir_name: str = RESULTS_DIR_NAME,
    dataset_start_idx: int = 0,
    nr_of_seeds: int = 10,
    methods: list[Method] = METHODS,
    max_n_evals: int = 10000,
    mean_range: tuple[float] = [0, 1],
    name_to_metric: dict[str, Callable] = {
        "KL": kl_div,
        "JS": js_div,
    },
) -> None:
    os.makedirs(f"{results_dir_name}/{nr_of_components}", exist_ok=True)
    rows = []
    existing_results = (
        pd.read_csv(f"{results_dir_name}/{nr_of_components}/results.csv", index_col=0)
        if os.path.exists(f"{results_dir_name}/{nr_of_components}/results.csv")
        else None
    )
    for mixture_idx, (parameters, dataset) in enumerate(
        generate_mixture_parameters(
            n_mixtures=nr_of_datasets,
            n_components=nr_of_components,
            mean_range=mean_range,
            min_overlap_value=min_overlap_value,
            max_overlap_value=max_overlap_value,
        )
    ):
        dataset_idx = dataset_start_idx + mixture_idx
        np.save(
            f"{results_dir_name}/{nr_of_components}/dataset_{dataset_idx}.npy", dataset
        )
        np.save(
            f"{results_dir_name}/{nr_of_components}/parameters_{dataset_idx}.npy",
            parameters,
        )
        dataset_start = perf_counter()
        for random_state in range(1, nr_of_seeds + 1):
            for method in methods:
                start = perf_counter()
                problem = GaussianMixtureProblem(
                    dataset,
                    nr_of_components,
                    bin_type=method.bin_type,
                    bin_number_method=method.bin_number_method,
                )
                solution = method.solver(
                    problem, max_n_evals=max_n_evals, random_state=random_state
                )
                time = perf_counter() - start
                row = {
                    "method": method.name,
                    "solution": solution.genome,
                    "time": time,
                    "fitness": solution.fitness,
                    "optimal_solution": parameters,
                    "optimal_fitness": problem(parameters),
                    "nr_of_components": nr_of_components,
                    "dataset_idx": dataset_idx,
                    "random_state": random_state,
                } | {
                    name: metric(parameters, solution.genome)
                    for name, metric in name_to_metric.items()
                }
                if time > 10:
                    print(
                        f"Nr of components {nr_of_components} dataset {dataset_idx} seed {random_state} took {time} seconds"
                    )
                rows.append(row)
        print(f"Dataset {dataset_idx} took {perf_counter() - dataset_start} seconds")
        all_results = (
            pd.concat([existing_results, pd.DataFrame(rows)])
            if existing_results is not None
            else pd.DataFrame(rows)
        )
        all_results.to_csv(f"{results_dir_name}/{nr_of_components}/results.csv")


def run_experiment_for_difficult_examples(
    nr_of_components: int,
    nr_of_datasets: int,
    results_dir_name: str = RESULTS_DIR_NAME,
    dataset_start_idx: int = 0,
    nr_of_seeds: int = 10,
    methods: list[Method] = METHODS,
    max_n_evals: int = 10000,
    name_to_metric: dict[str, Callable] = {
        "KL": kl_div,
        "JS": js_div,
    },
) -> None:
    os.makedirs(f"{results_dir_name}/{nr_of_components}", exist_ok=True)
    rows = []
    existing_results = (
        pd.read_csv(f"{results_dir_name}/{nr_of_components}/results.csv", index_col=0)
        if os.path.exists(f"{results_dir_name}/{nr_of_components}/results.csv")
        else None
    )
    for mixture_idx, (parameters, dataset) in enumerate(
        generate_difficult_mixture_parameters(n_mixtures=nr_of_datasets)
    ):
        dataset_idx = dataset_start_idx + mixture_idx
        np.save(
            f"{results_dir_name}/{nr_of_components}/dataset_{dataset_idx}.npy",
            dataset,
        )
        np.save(
            f"{results_dir_name}/{nr_of_components}/parameters_{dataset_idx}.npy",
            parameters,
        )
        dataset_start = perf_counter()
        for random_state in range(1, nr_of_seeds + 1):
            for method in methods:
                start = perf_counter()
                problem = GaussianMixtureProblem(
                    dataset,
                    nr_of_components,
                    bin_type=method.bin_type,
                    bin_number_method=method.bin_number_method,
                )
                solution = method.solver(
                    problem, max_n_evals=max_n_evals, random_state=random_state
                )
                time = perf_counter() - start
                row = {
                    "method": method.name,
                    "solution": solution.genome,
                    "time": time,
                    "fitness": solution.fitness,
                    "optimal_solution": parameters,
                    "optimal_fitness": problem(parameters),
                    "dataset_idx": dataset_idx,
                    "random_state": random_state,
                } | {
                    name: metric(parameters, solution.genome)
                    for name, metric in name_to_metric.items()
                }
                if time > 10:
                    print(
                        f"Dataset {dataset_idx} seed {random_state} took {time} seconds"
                    )
                rows.append(row)
        print(f"Dataset {dataset_idx} took {perf_counter() - dataset_start} seconds")
        all_results = (
            pd.concat([existing_results, pd.DataFrame(rows)])
            if existing_results is not None
            else pd.DataFrame(rows)
        )
        all_results.to_csv(f"{results_dir_name}/{nr_of_components}/results.csv")
