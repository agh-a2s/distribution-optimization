import sys

HMSPY_MODULE_PATH = "/Users/wojciechachtelik/Documents/studia/doktorat/pyhms"
if HMSPY_MODULE_PATH not in sys.path:
    sys.path.append(HMSPY_MODULE_PATH)

import optuna
from leap_ec.problem import FunctionProblem
from pyhms.config import TreeConfig, CMALevelConfig, EALevelConfig
from pyhms.tree import DemeTree
from pyhms.demes.single_pop_eas.sea import SimpleEA
from pyhms.sprout.sprout_mechanisms import SproutMechanism
from pyhms.sprout.sprout_filters import NBC_FarEnough, DemeLimit, LevelLimit
from pyhms.sprout.sprout_generators import NBC_Generator
from pyhms.stop_conditions.usc import dont_stop, metaepoch_limit
from pyhms.stop_conditions.gsc import singular_problem_eval_limit_reached
from pyhms.problem import EvalCutoffProblem
import pandas as pd
import numpy as np
from distribution_optimization_py.problem import ScaledGaussianMixtureProblem
from distribution_optimization_py.ga_style_operators import (
    ArithmeticCrossover,
    mutate_uniform,
)
import random
import leap_ec.ops as lops
from leap_ec.representation import Representation


def read_dataset_from_csv(path: str, column: str | None = "value") -> np.ndarray:
    return pd.read_csv(path)[column].values


DATASET_NAME = "chromatogram_time"
DATA = read_dataset_from_csv(f"./data/{DATASET_NAME}.csv")
NR_OF_MODES = 5
BUDGET = 5000
LOWER = 0
UPPER = 1
BOUNDS = np.array([(LOWER, UPPER)] * (NR_OF_MODES * 3 - 1))


class GAStyleEA(SimpleEA):
    """
    An implementation of SEA using LEAP.
    """

    def __init__(
        self,
        generations,
        problem,
        bounds,
        pop_size,
        mutation_std=1.0,
        k_elites=1,
        representation=None,
        p_mutation=1,
        p_crossover=1,
    ) -> None:
        super().__init__(
            generations,
            problem,
            bounds,
            pop_size,
            pipeline=[
                lops.tournament_selection,
                lops.clone,
                ArithmeticCrossover(
                    p_xover=p_crossover,
                ),
                mutate_uniform(
                    bounds=bounds,
                    p_mutate=p_mutation,
                ),
                lops.evaluate,
                lops.pool(size=pop_size),
            ],
            k_elites=k_elites,
            representation=representation,
        )

    @classmethod
    def create(cls, generations, problem, bounds, pop_size, **kwargs):
        mutation_std = kwargs.get("mutation_std") or 1.0
        k_elites = kwargs.get("k_elites") or 1
        p_mutation = kwargs.get("p_mutation") or 0.9
        p_crossover = kwargs.get("p_crossover") or 0.9
        return cls(
            generations=generations,
            problem=problem,
            bounds=bounds,
            pop_size=pop_size,
            mutation_std=mutation_std,
            k_elites=k_elites,
            p_mutation=p_mutation,
            p_crossover=p_crossover,
        )


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=f"{DATASET_NAME}_{BUDGET}_ea_cma_es_scaled",
        storage="sqlite:///example.db",
        direction="minimize",
    )

    def ea_cma_es_objective(trial):
        nbc_cut = trial.suggest_float("nbc_cut", 1.5, 4.0)
        nbc_trunc = trial.suggest_float("nbc_trunc", 0.1, 0.9)
        nbc_far = trial.suggest_float("nbc_far", 1.5, 4.0)
        level_limit = trial.suggest_int("level_limit", 2, 10)
        pop1 = trial.suggest_int("pop1", 20, 100)
        p_mutation1 = trial.suggest_float("p_mutation1", 0.0, 1.0)
        p_crossover1 = trial.suggest_float("p_crossover1", 0.0, 1.0)
        k_elites1 = trial.suggest_int("k_elites1", 1, 5)
        generations1 = trial.suggest_int("generations1", 2, 10)
        generations2 = trial.suggest_int("generations2", 3, 30)
        metaepoch2 = trial.suggest_int("metaepoch2", 10, 50)
        sigma2 = trial.suggest_float("sigma2", 0.1, 3.0)

        solutions = []
        for seed in range(1, 11):
            np.random.seed(seed)
            random.seed(seed)
            options = {"random_seed": seed}
            problem = ScaledGaussianMixtureProblem(DATA, NR_OF_MODES)
            function_problem = FunctionProblem(problem, maximize=False)
            cutoff_problem = EvalCutoffProblem(function_problem, BUDGET)
            config = [
                EALevelConfig(
                    ea_class=GAStyleEA,
                    generations=generations1,
                    problem=cutoff_problem,
                    bounds=BOUNDS,
                    pop_size=pop1,
                    lsc=dont_stop(),
                    k_elites=k_elites1,
                    representation=Representation(initialize=problem.initialize),
                    p_mutation=p_mutation1,
                    p_crossover=p_crossover1,
                ),
                CMALevelConfig(
                    problem=cutoff_problem,
                    bounds=BOUNDS,
                    lsc=metaepoch_limit(metaepoch2),
                    sigma0=sigma2,
                    generations=generations2,
                    random_state=seed,
                ),
            ]

            sprout = SproutMechanism(
                NBC_Generator(nbc_cut, nbc_trunc),
                [NBC_FarEnough(nbc_far, 2), DemeLimit(1)],
                [LevelLimit(level_limit)],
            )
            gsc = singular_problem_eval_limit_reached(BUDGET)
            tree_config = TreeConfig(config, gsc, sprout, options)
            deme_tree = DemeTree(tree_config)
            deme_tree.run()
            best_solution = deme_tree.best_individual
            solutions.append(best_solution.fitness)
        return np.mean(solutions)

    study.optimize(ea_cma_es_objective, n_trials=100)
