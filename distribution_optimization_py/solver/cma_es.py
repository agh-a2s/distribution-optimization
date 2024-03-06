import random

import numpy as np
from cma import CMAEvolutionStrategy

from ..problem import ScaledGaussianMixtureProblem
from .protocol import Solver


class CMAESSolver(Solver):
    def __call__(
        self,
        problem: ScaledGaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)
        x0 = problem.initialize()
        sigma = 1.0
        cma = CMAEvolutionStrategy(
            x0,
            sigma,
            inopts={
                "bounds": [problem.lower, problem.upper],
                "verbose": -9,
                "seed": random_state,
                "maxfevals": max_n_evals,
            },
        )
        cma.optimize(problem)
        return cma.result.xbest
