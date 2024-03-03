from typing import Protocol

import numpy as np

from ..problem import ScaledGaussianMixtureProblem


class Solver(Protocol):
    def __call__(
        self,
        problem: ScaledGaussianMixtureProblem,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        ...
