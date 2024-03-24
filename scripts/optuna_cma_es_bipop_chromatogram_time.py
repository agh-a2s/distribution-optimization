import random

import numpy as np
import optuna
import pandas as pd
from cma import fmin
from distribution_optimization_py.problem import ScaledGaussianMixtureProblem


def read_dataset_from_csv(path: str, column: str | None = "value") -> np.ndarray:
    return pd.read_csv(path)[column].values


DATASET_NAME = "chromatogram_time"
DATA = read_dataset_from_csv(f"./data/{DATASET_NAME}.csv")
NR_OF_MODES = 5
BUDGET = 5000


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=f"{DATASET_NAME}_{BUDGET}_cma_es_bipop_improved",
        storage="sqlite:///example.db",
        direction="minimize",
    )

    def cma_es_objective(trial):
        popsize = trial.suggest_int("popsize", 10, 100)
        sigma0 = trial.suggest_float("sigma0", 0.5, 7.5)
        incpopsize = trial.suggest_int("incpopsize", 1, 5)
        tolfun = trial.suggest_float("tolfun", 1e-15, 1e-8)
        fitness_values = []
        for seed in range(1, 11):
            np.random.seed(seed)
            random.seed(seed)
            problem = ScaledGaussianMixtureProblem(DATA, NR_OF_MODES)
            start = problem.initialize_warm_start()
            options = {
                "bounds": [problem.lower, problem.upper],
                "verbose": -9,
                "maxfevals": BUDGET,
                "popsize": popsize,
                "tolfun": tolfun,
            }

            res = fmin(
                problem,
                start,
                sigma0,
                options=options,
                bipop=True,
                restart_from_best=True,
                restarts=10,
                incpopsize=incpopsize,
            )
            fitness_values.append(res[1])
        return np.mean(fitness_values)

    study.optimize(cma_es_objective, n_trials=100)
