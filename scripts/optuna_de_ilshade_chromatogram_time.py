import random

import numpy as np
import optuna
import pandas as pd
import pyade.ilshade
from distribution_optimization_py.problem import ScaledGaussianMixtureProblem


def read_dataset_from_csv(path: str, column: str | None = "value") -> np.ndarray:
    return pd.read_csv(path)[column].values


DATASET_NAME = "chromatogram_time"
DATA = read_dataset_from_csv(f"./data/{DATASET_NAME}.csv")
NR_OF_MODES = 5
BUDGET = 5000


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=f"{DATASET_NAME}_{BUDGET}_de_ilshade_improved",
        storage="sqlite:///example.db",
        direction="minimize",
    )

    def de_ilshade_objective(trial):
        population_size = trial.suggest_int("population_size", 10, 300)
        memory_size = trial.suggest_int("memory_size", 1, 20)
        fitness_values = []
        for seed in range(1, 11):
            np.random.seed(seed)
            random.seed(seed)
            problem = ScaledGaussianMixtureProblem(DATA, NR_OF_MODES)
            algorithm = pyade.ilshade
            params = algorithm.get_default_params(dim=problem.lower.shape[0])
            params["bounds"] = np.column_stack([problem.lower, problem.upper])
            params["func"] = problem
            params["max_evals"] = BUDGET
            params["seed"] = seed
            params["population_size"] = population_size
            params["memory_size"] = memory_size
            _, fitness = algorithm.apply(**params)
            fitness_values.append(fitness)
        return np.mean(fitness_values)

    study.optimize(de_ilshade_objective, n_trials=100)
