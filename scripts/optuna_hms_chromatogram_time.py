import random

import numpy as np
import optuna
import pandas as pd
from distribution_optimization_py.ga_style_operators import GAStyleEA
from distribution_optimization_py.problem import ScaledGaussianMixtureProblem
from leap_ec.problem import FunctionProblem
from leap_ec.representation import Representation
from pyhms.config import CMALevelConfig, EALevelConfig, TreeConfig
from pyhms.problem import EvalCutoffProblem
from pyhms.sprout.sprout_filters import DemeLimit, LevelLimit, NBC_FarEnough
from pyhms.sprout.sprout_generators import NBC_Generator
from pyhms.sprout.sprout_mechanisms import SproutMechanism
from pyhms.stop_conditions.gsc import SingularProblemEvalLimitReached
from pyhms.stop_conditions.usc import DontStop, MetaepochLimit
from pyhms.tree import DemeTree


def read_dataset_from_csv(path: str, column: str | None = "value") -> np.ndarray:
    return pd.read_csv(path)[column].values


DATASET_NAME = "chromatogram_time"
DATA = read_dataset_from_csv(f"./data/{DATASET_NAME}.csv")
NR_OF_MODES = 5
BUDGET = 5000


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=f"{DATASET_NAME}_{BUDGET}_hms_ea_cma_es",
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
        use_warm_start = trial.suggest_categorical("use_warm_start", [True, False])

        solutions = []
        for seed in range(1, 11):
            np.random.seed(seed)
            random.seed(seed)
            options = {"random_seed": seed}
            problem = ScaledGaussianMixtureProblem(DATA, NR_OF_MODES)
            bounds = np.array([[lower, upper] for lower, upper in zip(problem.lower, problem.upper)])
            function_problem = FunctionProblem(problem, maximize=False)
            cutoff_problem = EvalCutoffProblem(function_problem, BUDGET)
            config = [
                EALevelConfig(
                    ea_class=GAStyleEA,
                    generations=generations1,
                    problem=cutoff_problem,
                    bounds=bounds,
                    pop_size=pop1,
                    lsc=DontStop(),
                    k_elites=k_elites1,
                    representation=Representation(initialize=problem.initialize),
                    p_mutation=p_mutation1,
                    p_crossover=p_crossover1,
                    use_warm_start=use_warm_start,
                ),
                CMALevelConfig(
                    problem=cutoff_problem,
                    bounds=bounds,
                    lsc=MetaepochLimit(metaepoch2),
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
            gsc = SingularProblemEvalLimitReached(BUDGET)
            tree_config = TreeConfig(config, gsc, sprout, options)
            deme_tree = DemeTree(tree_config)
            deme_tree.run()
            best_solution = deme_tree.best_individual
            solutions.append(best_solution.fitness)
        return np.mean(solutions)

    study.optimize(ea_cma_es_objective, n_trials=100)
