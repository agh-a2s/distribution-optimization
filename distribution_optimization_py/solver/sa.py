import random

import numpy as np
from leap_ec.representation import Representation
from pyhms.core.problem import FunctionProblem
from pyhms.core.individual import Individual
from ..evaluator.problem_wrapper import ProblemMonitor
from ..ga_style_operators import GAStyleEA, GAStyleWithEM
from ..problem import GaussianMixtureProblem, ScaledGaussianMixtureProblem
from .protocol import Solution, Solver
from .em import run_em_step
from scipy.optimize import dual_annealing


class SASolver(Solver):

    def __call__(
        self,
        problem: GaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ):
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)

        bounds = [(lower, upper) for lower, upper in zip(problem.lower, problem.upper)]

        result = dual_annealing(
            problem,
            x0=problem.initialize(),
            bounds=bounds,
            maxfun=max_n_evals,  # Equivalent to the maximum number of evaluations
            seed=random_state,
        )

        best_solution_genome = result.x

        scaled_genome = (
            problem.reals_to_internal(best_solution_genome)
            if isinstance(problem, ScaledGaussianMixtureProblem)
            else None
        )

        # Returning the final solution
        return Solution(
            fitness=problem(best_solution_genome),
            genome=best_solution_genome,
            scaled_genome=scaled_genome,
            log_likelihood=problem.log_likelihood(best_solution_genome),
            fitness_values=np.array([]),
        )
