import random

import numpy as np
import pyade.ilshade

from ..problem import ScaledGaussianMixtureProblem
from .protocol import Solver


class DESolver(Solver):
    def __call__(
        self,
        problem: ScaledGaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)
        algorithm = pyade.ilshade
        params = algorithm.get_default_params(dim=problem.lower.shape[0])
        params["bounds"] = np.column_stack([problem.lower, problem.upper])
        params["func"] = problem
        params["max_evals"] = max_n_evals
        solution, fitness = algorithm.apply(**params)
        return solution
