import random

import numpy as np
from leap_ec.problem import FunctionProblem
from leap_ec.representation import Representation
from pyhms.problem import EvalCutoffProblem

from ..ga_style_operators import GAStyleEA
from ..problem import ScaledGaussianMixtureProblem
from .protocol import Solver

GA_CONFIG = {"k_elites": 5, "p_mutation": 0.8, "p_crossover": 0.9, "pop_size": 50}


class GASolver(Solver):
    def __call__(
        self,
        problem: ScaledGaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ):
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)
        bounds = np.column_stack((problem.lower, problem.upper))
        function_problem = FunctionProblem(problem, maximize=False)
        eval_cutoff_problem = EvalCutoffProblem(function_problem, max_n_evals)
        ea = GAStyleEA(
            generations=1,
            problem=eval_cutoff_problem,
            bounds=bounds,
            pop_size=GA_CONFIG["pop_size"],
            k_elites=GA_CONFIG["k_elites"],
            p_mutation=GA_CONFIG["p_mutation"],
            p_crossover=GA_CONFIG["p_crossover"],
            representation=Representation(initialize=lambda: problem.initialize()),
        )
        all_populations = []
        population = None
        iterations_count = int(max_n_evals / GA_CONFIG["pop_size"])
        for _ in range(iterations_count):
            population = ea.run(population)
            all_populations.extend(population)
        return max(all_populations).genome
